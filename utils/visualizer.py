import plotly.graph_objects as go
import streamlit as st

def plot_signals(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlestick'))
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == 0]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers',
                             marker=dict(color='green', size=8), name='Buy Call'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers',
                             marker=dict(color='red', size=8), name='Buy Put'))
    st.plotly_chart(fig)