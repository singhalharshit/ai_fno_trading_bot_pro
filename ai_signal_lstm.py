# ai_signal_lstm.py

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")


def fetch_price_data(symbol, period="60d", interval="30m"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    return df[['Close']]


def create_sequences(data, sequence_length=50):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(1 if data[i] > data[i-1] else 0)
    return np.array(X), np.array(y)


def train_lstm_model(symbol):
    print(f"🔁 Training LSTM model for: {symbol}")
    df = fetch_price_data(symbol)
    if df.empty or len(df) < 100:
        print(f"❌ Insufficient data for {symbol}")
        return None, None, None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])

    X, y = create_sequences(scaled_data, sequence_length=50)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    accuracy = model.evaluate(X, y, verbose=0)[1]
    print(f"✅ LSTM model trained with accuracy: {accuracy:.2f}")
    return model, scaler, df


def predict_lstm_signal(model, scaler, df):
    latest_data = df[-50:][['Close']]
    scaled = scaler.transform(latest_data)
    X_input = np.reshape(scaled, (1, scaled.shape[0], 1))
    pred = model.predict(X_input)[0][0]
    return "BUY CALL" if pred > 0.5 else "BUY PUT"


# Quick test
if __name__ == "__main__":
    test_symbol = "RELIANCE.NS"
    model, scaler, df = train_lstm_model(test_symbol)
    if model and scaler:
        decision = predict_lstm_signal(model, scaler, df)
        print(f"🧠 AI Signal for {test_symbol}: {decision}")