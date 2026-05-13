# -*- coding: utf-8 -*-
import itertools
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DEFAULT_DB_PATH = Path(__file__).with_name("shioaji_sample.db")


st.set_page_config(page_title="程式交易策略回測 APP", layout="wide")


def find_default_db() -> str:
    if DEFAULT_DB_PATH.exists():
        return str(DEFAULT_DB_PATH)
    local_db = Path(__file__).with_name("shioaji.db")
    if local_db.exists():
        return str(local_db)
    return str(DEFAULT_DB_PATH)


@st.cache_data(show_spinner=False)
def list_tables(db_path: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "select name from sqlite_master where type='table' order by name"
        ).fetchall()
    return [row[0] for row in rows]


@st.cache_data(show_spinner=False)
def table_date_range(db_path: str, table: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f'select min(time), max(time) from "{table}"').fetchone()
    return pd.to_datetime(row[0]), pd.to_datetime(row[1])


@st.cache_data(show_spinner=False)
def load_kbars(
    db_path: str,
    table: str,
    start_date: str,
    end_date: str,
    cycle_minutes: int,
) -> pd.DataFrame:
    query = f'''
        select time, open, high, low, close, volume, amount, product
        from "{table}"
        where time >= ? and time <= ?
        order by time
    '''
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df.sort_values("time").set_index("time")
    product = str(df["product"].dropna().iloc[0]) if "product" in df and df["product"].notna().any() else table

    if cycle_minutes > 1:
        df = df.resample(f"{cycle_minutes}min").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
        )
        df = df.dropna(subset=["open", "high", "low", "close"])

    df["product"] = product
    return df.reset_index()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    out["ma_short"] = close.rolling(p["ma_short"]).mean()
    out["ma_long"] = close.rolling(p["ma_long"]).mean()

    out["rsi_short"] = rsi(close, p["rsi_short"])
    out["rsi_long"] = rsi(close, p["rsi_long"])

    mid = close.rolling(p["bb_period"]).mean()
    std = close.rolling(p["bb_period"]).std()
    out["bb_mid"] = mid
    out["bb_upper"] = mid + p["bb_std"] * std
    out["bb_lower"] = mid - p["bb_std"] * std

    ema_fast = close.ewm(span=p["macd_fast"], adjust=False).mean()
    ema_slow = close.ewm(span=p["macd_slow"], adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=p["macd_signal"], adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    lowest = low.rolling(p["kdj_period"]).min()
    highest = high.rolling(p["kdj_period"]).max()
    rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    out["k"] = rsv.ewm(alpha=1 / p["kdj_k"], adjust=False).mean()
    out["d"] = out["k"].ewm(alpha=1 / p["kdj_d"], adjust=False).mean()
    out["j"] = 3 * out["k"] - 2 * out["d"]
    return out


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)


def make_signal(df: pd.DataFrame, strategy: str) -> pd.Series:
    signal = pd.Series(0, index=df.index, dtype=int)

    if strategy == "MA 移動平均線":
        signal[crossover(df["ma_short"], df["ma_long"])] = 1
        signal[crossunder(df["ma_short"], df["ma_long"])] = -1
    elif strategy == "RSI 順勢":
        signal[crossover(df["rsi_short"], df["rsi_long"]) & (df["rsi_long"] > 50)] = 1
        signal[crossunder(df["rsi_short"], df["rsi_long"]) & (df["rsi_long"] < 50)] = -1
    elif strategy == "RSI 逆勢":
        signal[(df["rsi_short"].shift(1) < 30) & (df["rsi_short"] >= 30)] = 1
        signal[(df["rsi_short"].shift(1) > 70) & (df["rsi_short"] <= 70)] = -1
    elif strategy == "布林通道":
        signal[(df["close"].shift(1) <= df["bb_lower"].shift(1)) & (df["close"] > df["bb_lower"])] = 1
        signal[(df["close"].shift(1) >= df["bb_upper"].shift(1)) & (df["close"] < df["bb_upper"])] = -1
    elif strategy == "MACD":
        signal[crossover(df["macd"], df["macd_signal"]) & (df["macd"] > 0)] = 1
        signal[crossunder(df["macd"], df["macd_signal"]) & (df["macd"] < 0)] = -1
    elif strategy == "KDJ":
        signal[crossover(df["k"], df["d"]) & (df["j"] < 30)] = 1
        signal[crossunder(df["k"], df["d"]) & (df["j"] > 70)] = -1
    elif strategy == "自訂策略: MACD + KDJ":
        signal[
            crossover(df["macd"], df["macd_signal"])
            & (df["macd_hist"] > 0)
            & crossover(df["k"], df["d"])
        ] = 1
        signal[
            crossunder(df["macd"], df["macd_signal"])
            & (df["macd_hist"] < 0)
            & crossunder(df["k"], df["d"])
        ] = -1

    return signal


def backtest(df: pd.DataFrame, strategy: str, p: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = add_indicators(df, p).copy()
    data["signal"] = make_signal(data, strategy)

    position = 0
    entry_price = 0.0
    entry_time = None
    stop_price = np.nan
    trades = []
    equity = []
    capital = 0.0

    for i in range(1, len(data) - 1):
        row = data.iloc[i]
        next_row = data.iloc[i + 1]
        sig = int(row["signal"])

        if position == 0 and sig != 0:
            position = sig
            entry_price = float(next_row["open"])
            entry_time = next_row["time"]
            stop_price = entry_price - p["stop_loss"] if position > 0 else entry_price + p["stop_loss"]
            continue

        if position == 0:
            equity.append(capital)
            continue

        if position > 0:
            stop_price = max(stop_price, float(row["close"]) - p["stop_loss"])
            should_exit = sig == -1 or float(row["close"]) < stop_price
        else:
            stop_price = min(stop_price, float(row["close"]) + p["stop_loss"])
            should_exit = sig == 1 or float(row["close"]) > stop_price

        if should_exit:
            exit_price = float(next_row["open"])
            qty = int(p["qty"])
            profit = (exit_price - entry_price) * qty * position
            capital += profit
            trades.append(
                {
                    "方向": "做多" if position > 0 else "做空",
                    "進場時間": entry_time,
                    "進場價": entry_price,
                    "出場時間": next_row["time"],
                    "出場價": exit_price,
                    "數量": qty,
                    "損益": profit,
                    "報酬率": profit / abs(entry_price * qty) if entry_price else 0,
                }
            )
            position = 0
            entry_price = 0.0
            entry_time = None
            stop_price = np.nan

        equity.append(capital)

    if position != 0 and len(data) > 0:
        last = data.iloc[-1]
        qty = int(p["qty"])
        exit_price = float(last["close"])
        profit = (exit_price - entry_price) * qty * position
        capital += profit
        trades.append(
            {
                "方向": "做多" if position > 0 else "做空",
                "進場時間": entry_time,
                "進場價": entry_price,
                "出場時間": last["time"],
                "出場價": exit_price,
                "數量": qty,
                "損益": profit,
                "報酬率": profit / abs(entry_price * qty) if entry_price else 0,
            }
        )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        metrics = {
            "總損益": 0.0,
            "交易次數": 0,
            "勝率": 0.0,
            "平均每筆損益": 0.0,
            "最大回撤": 0.0,
            "報酬回撤比": 0.0,
            "Profit Factor": 0.0,
            "最佳化分數": -999999.0,
        }
    else:
        profits = trades_df["損益"].astype(float)
        eq = profits.cumsum()
        drawdown = eq.cummax() - eq
        gross_profit = profits[profits > 0].sum()
        gross_loss = abs(profits[profits < 0].sum())
        mdd = float(drawdown.max())
        total = float(profits.sum())
        metrics = {
            "總損益": total,
            "交易次數": int(len(trades_df)),
            "勝率": float((profits > 0).mean()),
            "平均每筆損益": float(profits.mean()),
            "最大回撤": mdd,
            "報酬回撤比": float(total / mdd) if mdd > 0 else float(total),
            "Profit Factor": float(gross_profit / gross_loss) if gross_loss > 0 else float(gross_profit),
            "最佳化分數": float(total - p["risk_weight"] * mdd + 100 * (profits > 0).mean()),
        }

    data["equity"] = pd.Series(equity, index=data.index[: len(equity)]).reindex(data.index).ffill().fillna(0)
    return data, trades_df, metrics


def optimize(df: pd.DataFrame, strategy: str, base: dict) -> pd.DataFrame:
    grids = {
        "MA 移動平均線": {
            "ma_short": [3, 5, 8],
            "ma_long": [15, 25, 40],
            "stop_loss": [base["stop_loss"], base["stop_loss"] * 2],
        },
        "RSI 順勢": {
            "rsi_short": [5, 7, 10],
            "rsi_long": [14, 21, 28],
            "stop_loss": [base["stop_loss"], base["stop_loss"] * 2],
        },
        "RSI 逆勢": {
            "rsi_short": [5, 7, 10, 14],
            "stop_loss": [base["stop_loss"], base["stop_loss"] * 2],
        },
        "布林通道": {
            "bb_period": [15, 20, 30],
            "bb_std": [1.5, 2.0, 2.5],
            "stop_loss": [base["stop_loss"], base["stop_loss"] * 2],
        },
        "MACD": {
            "macd_fast": [8, 12],
            "macd_slow": [21, 26, 35],
            "macd_signal": [5, 9],
            "stop_loss": [base["stop_loss"], base["stop_loss"] * 2],
        },
        "KDJ": {
            "kdj_period": [9, 14, 21],
            "kdj_k": [3, 5],
            "kdj_d": [3, 5],
            "stop_loss": [base["stop_loss"], base["stop_loss"] * 2],
        },
        "自訂策略: MACD + KDJ": {
            "macd_fast": [8, 12],
            "macd_slow": [21, 26],
            "macd_signal": [5, 9],
            "kdj_period": [9, 14],
            "stop_loss": [base["stop_loss"], base["stop_loss"] * 2],
        },
    }
    grid = grids[strategy]
    rows = []
    keys = list(grid)
    for values in itertools.product(*[grid[k] for k in keys]):
        p = base.copy()
        p.update(dict(zip(keys, values)))
        if p["ma_short"] >= p["ma_long"] or p["rsi_short"] >= p["rsi_long"] or p["macd_fast"] >= p["macd_slow"]:
            continue
        _, _, metrics = backtest(df, strategy, p)
        rows.append({**{k: p[k] for k in keys}, **metrics})
    return pd.DataFrame(rows).sort_values("最佳化分數", ascending=False).reset_index(drop=True)


def ai_review(strategy: str, metrics: dict, trades: pd.DataFrame, opt: pd.DataFrame | None) -> str:
    total = metrics["總損益"]
    mdd = metrics["最大回撤"]
    win_rate = metrics["勝率"]
    trade_count = metrics["交易次數"]
    pf = metrics["Profit Factor"]

    if trade_count < 5:
        verdict = "交易次數偏少，績效代表性不足，建議拉長期間或改用較短 KBar 週期。"
    elif total > 0 and mdd > 0 and total / mdd >= 1.5 and win_rate >= 0.45:
        verdict = "整體表現較穩健，報酬能覆蓋主要回撤，適合作為候選策略。"
    elif total > 0:
        verdict = "策略有獲利，但風險控制仍要觀察，特別是最大回撤與連續虧損。"
    else:
        verdict = "目前參數組合不理想，策略在此資料區間未能產生正報酬。"

    best_text = ""
    if opt is not None and not opt.empty:
        best = opt.iloc[0]
        best_text = (
            f" 參數最佳化結果顯示，最佳分數組合的總損益為 {best['總損益']:.2f}，"
            f"最大回撤為 {best['最大回撤']:.2f}，勝率為 {best['勝率']:.2%}。"
        )

    risk_note = "最大回撤為 0，可能是交易太少或全程未發生回落。" if mdd == 0 else f"報酬回撤比為 {metrics['報酬回撤比']:.2f}。"
    return (
        f"AI 自動評估：{strategy} 在本次回測中共有 {trade_count} 筆交易，"
        f"總損益 {total:.2f}，勝率 {win_rate:.2%}，Profit Factor {pf:.2f}。"
        f"{risk_note} {verdict}{best_text}"
    )


def plot_chart(df: pd.DataFrame, trades: pd.DataFrame, strategy: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="KBar",
        )
    )

    overlays = {
        "MA 移動平均線": [("ma_short", "MA Short"), ("ma_long", "MA Long")],
        "布林通道": [("bb_upper", "Upper"), ("bb_mid", "Middle"), ("bb_lower", "Lower")],
        "MACD": [("macd", "DIF"), ("macd_signal", "DEA")],
        "KDJ": [("k", "K"), ("d", "D"), ("j", "J")],
        "自訂策略: MACD + KDJ": [("macd", "DIF"), ("macd_signal", "DEA")],
    }
    for col, name in overlays.get(strategy, []):
        if col in df:
            fig.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines", name=name))

    if not trades.empty:
        fig.add_trace(
            go.Scatter(
                x=trades["進場時間"],
                y=trades["進場價"],
                mode="markers",
                marker=dict(symbol="triangle-up", color="#d62728", size=10),
                name="進場",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trades["出場時間"],
                y=trades["出場價"],
                mode="markers",
                marker=dict(symbol="triangle-down", color="#1f77b4", size=10),
                name="出場",
            )
        )

    fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=35, b=10))
    return fig


st.title("程式交易策略回測與參數最佳化")

with st.sidebar:
    st.header("資料設定")
    db_path = st.text_input("SQLite 資料庫路徑", value=find_default_db())
    tables = list_tables(db_path) if Path(db_path).exists() else []
    if not tables:
        st.error("找不到資料表，請確認 shioaji.db 路徑。")
        st.stop()

    table = st.selectbox("選擇資料表", tables, index=tables.index("stock_KBar_2330") if "stock_KBar_2330" in tables else 0)
    min_dt, max_dt = table_date_range(db_path, table)
    start = st.date_input("開始日期", value=max(min_dt.date(), pd.Timestamp("2022-01-01").date()), min_value=min_dt.date(), max_value=max_dt.date())
    end = st.date_input("結束日期", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
    cycle = st.selectbox("KBar 週期", [1, 5, 15, 30, 60, 240, 1440, 2880], index=2, format_func=lambda x: f"{x} 分鐘")

    st.header("策略設定")
    strategy = st.selectbox(
        "策略",
        ["MA 移動平均線", "RSI 順勢", "RSI 逆勢", "布林通道", "MACD", "KDJ", "自訂策略: MACD + KDJ"],
    )
    qty = st.number_input("每次交易數量", min_value=1, value=1, step=1)
    stop_loss = st.number_input("移動停損點數", min_value=0.1, value=10.0, step=1.0)
    risk_weight = st.slider("最佳化風險權重", 0.0, 5.0, 1.0, 0.1)

    st.header("模型參數")
    params = {
        "ma_short": st.number_input("短 MA", 2, 100, 5),
        "ma_long": st.number_input("長 MA", 3, 300, 20),
        "rsi_short": st.number_input("短 RSI", 2, 100, 5),
        "rsi_long": st.number_input("長 RSI", 3, 200, 14),
        "bb_period": st.number_input("布林週期", 5, 200, 20),
        "bb_std": st.number_input("布林標準差倍數", 0.5, 5.0, 2.0, 0.1),
        "macd_fast": st.number_input("MACD Fast", 2, 100, 12),
        "macd_slow": st.number_input("MACD Slow", 3, 200, 26),
        "macd_signal": st.number_input("MACD Signal", 2, 100, 9),
        "kdj_period": st.number_input("KDJ RSV 週期", 3, 100, 9),
        "kdj_k": st.number_input("K 平滑", 1, 20, 3),
        "kdj_d": st.number_input("D 平滑", 1, 20, 3),
        "qty": qty,
        "stop_loss": float(stop_loss),
        "risk_weight": float(risk_weight),
    }
    run_opt = st.checkbox("執行參數最佳化", value=False)

start_sql = f"{start} 00:00:00"
end_sql = f"{end} 23:59:59"
source_df = load_kbars(db_path, table, start_sql, end_sql, int(cycle))
if source_df.empty:
    st.warning("這個日期區間沒有資料。")
    st.stop()

result_df, trades_df, metrics = backtest(source_df, strategy, params)
opt_df = optimize(source_df, strategy, params) if run_opt else None

cols = st.columns(6)
cols[0].metric("總損益", f"{metrics['總損益']:.2f}")
cols[1].metric("交易次數", metrics["交易次數"])
cols[2].metric("勝率", f"{metrics['勝率']:.2%}")
cols[3].metric("最大回撤", f"{metrics['最大回撤']:.2f}")
cols[4].metric("報酬回撤比", f"{metrics['報酬回撤比']:.2f}")
cols[5].metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")

st.plotly_chart(plot_chart(result_df, trades_df, strategy), use_container_width=True)

st.subheader("AI 自動績效評估")
st.write(ai_review(strategy, metrics, trades_df, opt_df))

tab1, tab2, tab3 = st.tabs(["交易紀錄", "權益曲線", "參數最佳化"])
with tab1:
    st.dataframe(trades_df, use_container_width=True)
with tab2:
    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(x=result_df["time"], y=result_df["equity"], mode="lines", name="累計損益"))
    equity_fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(equity_fig, use_container_width=True)
with tab3:
    if opt_df is None:
        st.info("勾選左側「執行參數最佳化」後會產生結果。最佳化分數同時考慮總損益、最大回撤與勝率。")
    else:
        st.dataframe(opt_df.head(30), use_container_width=True)
