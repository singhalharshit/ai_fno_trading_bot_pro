import yfinance as yf

def get_intraday_data(symbol, interval="15m", days=5):
    try:
        df = yf.download(tickers=symbol, period=f"{days}d", interval=interval)
        if df.empty:
            # Fallback: try daily instead of intraday
            df = yf.download(tickers=symbol, period="30d", interval="1d")
        return df
    except Exception as e:
        print(f"⚠️ Error fetching {symbol}: {e}")
        return None