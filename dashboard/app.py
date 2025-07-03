# dashboard/app.py

import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from symbol_screener_ai import get_ai_screened_symbols
from ai_signal_lstm import train_lstm_model, predict_lstm_signal

st.set_page_config(page_title="AI F&O Bot Dashboard", layout="wide")
st.title("🤖 AI F&O Trading Bot - Dashboard")

st.markdown("""
This dashboard shows live symbol screening, AI model accuracy, and signal predictions.
Run this during live market hours (Mon–Fri 9:15am–3:30pm).
""")

with st.spinner("🔍 Running Screener & AI Models..."):
    symbols, _ = get_ai_screened_symbols()
    results = []

    for symbol in symbols:
        model, scaler, df = train_lstm_model(symbol)
        if model:
            signal = predict_lstm_signal(model, scaler, df)
            acc = model.evaluate(*model.validation_data, verbose=0)[1] if hasattr(model, 'validation_data') else 0.0
            results.append({
                'Symbol': symbol,
                'Signal': signal,
                'Last Price': df['Close'].iloc[-1],
                'Accuracy': round(acc, 3)
            })

if not results:
    st.error("⚠️ No data available. Try during market hours.")
else:
    df_result = pd.DataFrame(results)
    st.dataframe(df_result)

    top = df_result.sort_values("Accuracy", ascending=False).head(1)
    st.success(f"Top Symbol to Watch: **{top['Symbol'].values[0]}** → Signal: **{top['Signal'].values[0]}**")
