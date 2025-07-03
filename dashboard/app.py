# dashboard/app.py

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fetch_data import get_intraday_data
from models.model_xgb import FNOSmartAI
from models.model_lstm import LSTMTrader
from rl.reinforce_ai import RLTrader
from simulator.paper_trader import PaperTrader
from rl.self_learning import TradeMemory
from symbol_screener_ai import get_ai_screened_symbols

st.set_page_config(page_title="AI F&O Bot Dashboard", layout="wide")
st.title("📊 AI F&O Trading Bot Dashboard")

model_type = st.selectbox("Select AI Model", ["XGBoost", "LSTM", "Reinforcement Learning"])
mode = st.selectbox("Mode", ["paper", "live"])

symbols, _ = get_ai_screened_symbols()
selected_symbol = st.selectbox("Select Symbol", symbols)

if st.button("🔍 Run Analysis"):
    df = get_intraday_data(selected_symbol)
    if df.empty:
        st.error("No data found for selected symbol")
    else:
        if model_type == "XGBoost":
            model = FNOSmartAI()
        elif model_type == "LSTM":
            model = LSTMTrader()
        else:
            model = RLTrader()

        acc = model.train(df)
        signal = model.predict(df)
        trade_type = "BUY CALL" if signal == 1 else "BUY PUT"

        price = df['Close'].iloc[-1]
        trader = PaperTrader() if mode == "paper" else None
        trader.place_trade(trade_type, price, selected_symbol)

        st.metric("📌 Trade Type", trade_type)
        st.metric("✅ Accuracy", f"{acc:.2f}")
        st.metric("📈 Current Price", f"{price:.2f}")

        st.subheader("Trade Log")
        st.dataframe(pd.DataFrame(trader.get_trade_log()))
