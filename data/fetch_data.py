import yfinance as yf
import pandas as pd

def get_intraday_data(symbol: str, interval: str = "5m", days: int = 5):
    df = yf.download(tickers=symbol, period=f"{days}d", interval=interval, auto_adjust=False)
    df.dropna(inplace=True)
    df['RSI'] = df['Close'].rolling(14).apply(lambda x: (x.diff() > 0).sum() / 14 * 100)
    df['EMA_5'] = df['Close'].ewm(span=5).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
    df.dropna(inplace=True)
    return df