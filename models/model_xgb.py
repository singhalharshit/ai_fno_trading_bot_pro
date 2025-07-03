from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FNOSmartAI:
    def __init__(self):
        self.model = XGBClassifier()

    def train(self, df):
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train)
        self.acc = accuracy_score(y_test, self.model.predict(X_test))
        return self.acc

    def predict(self, df):
        latest = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[[-1]]
        return self.model.predict(latest)[0]

    def annotate(self, df):
        df['signal'] = self.model.predict(df[['Open', 'High', 'Low', 'Close', 'Volume']])
        return df