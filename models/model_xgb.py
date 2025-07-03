from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FNOSmartAI:
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def preprocess(self, df):
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA_5', 'EMA_20', 'MACD', 'Signal_Line']
        X = df[features]
        y = df['target']
        return train_test_split(X, y, test_size=0.2, random_state=42), features

    def train(self, df):
        (X_train, X_test, y_train, y_test), self.features = self.preprocess(df)
        self.model.fit(X_train, y_train)
        acc = accuracy_score(y_test, self.model.predict(X_test))
        return acc

    def predict(self, df):
        latest = df[self.features].iloc[-1]
        return self.model.predict([latest])[0]