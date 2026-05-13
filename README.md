# 程式交易策略回測 Streamlit APP

本專案為期末報告使用，資料來源為老師提供的 `shioaji.db`，並匯出部分 KBar 資料成 `shioaji_sample.db` 供 Streamlit Cloud 部署。

## 功能

- 讀取 SQLite KBar 資料
- 支援 MA、RSI 順勢、RSI 逆勢、布林通道、MACD、KDJ 策略
- 額外提供自訂策略：MACD + KDJ
- 顯示 K 線圖、進出場點、交易紀錄、累計損益
- 提供參數最佳化，最佳化分數同時考慮報酬、最大回撤與勝率
- 以生成式 AI 風格文字自動評估策略績效與風險

## Streamlit Cloud 設定

- Main file path: `app.py`
- Python dependencies: `requirements.txt`
- Database file: `shioaji_sample.db`
