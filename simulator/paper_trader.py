class PaperTrader:
    def __init__(self):
        self.trades = []

    def place_trade(self, signal, price, symbol):
        self.trades.append({
            "symbol": symbol,
            "signal": signal,
            "price": price,
        })

    def get_trade_log(self):
        return self.trades