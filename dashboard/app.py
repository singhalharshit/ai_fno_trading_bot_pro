import streamlit as st
import pandas as pd
import sys
import os

# Add root path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fetch_data import get_intraday_data
from models.model_xgb import FNOSmartAI
from simulator.paper_trader import PaperTrader

st.title("AI F&O Trading Bot - Dashboard")

symbol = st.selectbox("Select Symbol", ["^NSEI", "RELIANCE.NS", "TCS.NS"])
df = get_intraday_data(symbol)

model = FNOSmartAI()
acc = model.train(df)
signal = model.predict(df)

trader = PaperTrader()
trade_type = "BUY CALL" if signal == 1 else "BUY PUT"
trader.place_trade(trade_type, float(df['Close'].iloc[-1]), symbol)

st.success(f"Model Accuracy: {acc:.2f}")
st.write("Last Signal:", trade_type)
st.dataframe(trader.get_trade_log())
