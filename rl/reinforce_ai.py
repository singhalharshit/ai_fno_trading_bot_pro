# rl/reinforce_ai.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random

class RLTrader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.q_table = {}  # state: [buy_call_q, buy_put_q]
        self.epsilon = 0.1  # Exploration factor
        self.alpha = 0.5    # Learning rate
        self.gamma = 0.9    # Discount factor

    def extract_features(self, df):
        df = df.copy()
        df['ma'] = df['Close'].rolling(window=5).mean()
        df['rsi'] = self._rsi(df['Close'])
        df.dropna(inplace=True)
        features = df[['Close', 'ma', 'rsi']].values
        return features, df

    def _rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _state(self, row):
        return tuple(np.round(row, 2))  # discretize for Q-table

    def train(self, df):
        X, df = self.extract_features(df)
        y = np.sign(np.diff(df['Close'].values, prepend=df['Close'].iloc[0]))
        y = np.where(y > 0, 1, 0)  # 1 for price up (call), 0 for down (put)

        self.scaler.fit(X)
        X = self.scaler.transform(X)

        for i in range(len(X)):
            state = self._state(X[i])
            if state not in self.q_table:
                self.q_table[state] = [0.0, 0.0]  # init Q-values

            action = y[i]  # supervised reward for this bar
            reward = 1 if y[i] == action else -1
            old_value = self.q_table[state][action]
            self.q_table[state][action] = old_value + self.alpha * (reward + self.gamma * max(self.q_table[state]) - old_value)

        return round(sum([max(q) for q in self.q_table.values()]) / len(self.q_table), 2)

    def predict(self, df):
        X, df = self.extract_features(df)
        X = self.scaler.transform(X)
        state = self._state(X[-1])

        if state not in self.q_table or random.random() < self.epsilon:
            return random.choice([0, 1])  # explore

        return int(np.argmax(self.q_table[state]))
