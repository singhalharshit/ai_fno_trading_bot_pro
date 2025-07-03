# symbol_screener_ai.py (Full AI-Driven)

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# 🔄 Dynamically fetch F&O universe (mocked for now, can be replaced with live NSE API scrape)
def get_dynamic_universe():
    return [
        "RELIANCE.NS", "INFY.NS", "ICICIBANK.NS", "HDFCBANK.NS", "SBIN.NS",
        "TCS.NS", "ITC.NS", "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "NESTLEIND.NS", "BAJFINANCE.NS", "SUNPHARMA.NS", "WIPRO.NS",
        "HINDUNILVR.NS", "HCLTECH.NS", "ADANIENT.NS", "TITAN.NS",
        "^NSEI", "^NSEBANK"
    ]

# ⛏️ Feature extraction from historical OHLCV + indicators
def extract_features(symbol):
    try:
        df = yf.download(symbol, period="15d", interval="1h", progress=False)
        if df.empty or len(df) < 50:
            # fallback to daily
            df = yf.download(symbol, period="30d", interval="1d", progress=False)
        if df.empty or len(df) < 50:
            return None

        df.dropna(inplace=True)
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['Signal'] = macd.macd_signal()
        df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(10).std()

        df.dropna(inplace=True)

        feature_df = df[['RSI', 'MACD', 'Signal', 'ATR', 'Volatility', 'Volume']].copy()
        feature_df['label'] = np.where(df['Close'].shift(-2) > df['Close'], 1, 0)  # future price gain label
        return feature_df

    except Exception as e:
        return None

# 🧠 Train AI model to predict trade-worthy symbols
def train_ai_model(dataframes):
    combined = pd.concat(dataframes, axis=0)
    combined.dropna(inplace=True)
    X = combined.drop('label', axis=1)
    y = combined['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(n_estimators=100))
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"✅ Screener AI Accuracy: {acc:.2f}")
    return pipe

# 🚀 Predict best symbols for today
def get_ai_screened_symbols():
    universe = get_dynamic_universe()
    feature_sets = []
    symbol_map = {}

    print("📡 Fetching data...")
    for sym in universe:
        feats = extract_features(sym)
        if feats is not None:
            feature_sets.append(feats)
            symbol_map[sym] = feats.iloc[-1][['RSI', 'MACD', 'Signal', 'ATR', 'Volatility', 'Volume']]

    if not feature_sets:
        print("❌ No data fetched.")
        return [], None

    model = train_ai_model(feature_sets)
    # Score today's symbol snapshot
    ranked = []
    for sym, features in symbol_map.items():
        X = pd.DataFrame([features.values], columns=features.index)
        pred_prob = model.predict_proba(X)[0][1]
        ranked.append((sym, pred_prob))

    ranked.sort(key=lambda x: x[1], reverse=True)
    top = [r[0] for r in ranked[:3]]
    print("🔥 Top trade symbols:", top)
    return top, model

# 🔎 If running directly
if __name__ == "__main__":
    selected, _ = get_ai_screened_symbols()
    print("Trade universe:", selected)
