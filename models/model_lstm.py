# models/model_lstm.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

class LSTMTrader:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.window_size = 10

    def prepare_data(self, df):
        df = df[['Close']].copy()
        df.dropna(inplace=True)
        scaled = self.scaler.fit_transform(df.values)

        X, y = [], []
        for i in range(self.window_size, len(scaled)):
            X.append(scaled[i - self.window_size:i, 0])
            y.append(1 if scaled[i, 0] > scaled[i-1, 0] else 0)

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def train(self, df):
        X, y = self.prepare_data(df)
        if len(X) == 0:
            return 0

        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        _, acc = self.model.evaluate(X, y, verbose=0)
        return round(acc, 2)

    def predict(self, df):
        X, _ = self.prepare_data(df)
        if len(X) == 0 or self.model is None:
            return 0

        pred = self.model.predict(X[-1].reshape(1, X.shape[1], 1), verbose=0)
        return int(pred[0][0] > 0.5)