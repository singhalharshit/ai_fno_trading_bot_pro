class PaperTrader:
    def __init__(self):
        self.log = []

    def place_trade(self, signal, price, symbol):
        self.log.append({'symbol': symbol, 'signal': signal, 'price': price})

    def get_trade_log(self):
        import pandas as pd
        return pd.DataFrame(self.log)