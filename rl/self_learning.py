# rl/self_learning.py

import json
import os

# Basic reinforcement store for learning trade outcomes
class TradeMemory:
    def __init__(self, filepath="rl/trade_memory.json"):
        self.filepath = filepath
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def record_trade(self, symbol, signal, result):
        if symbol not in self.data:
            self.data[symbol] = []
        self.data[symbol].append({"signal": signal, "result": result})
        self.save()

    def get_win_rate(self, symbol):
        if symbol not in self.data or len(self.data[symbol]) < 3:
            return 0.5  # neutral
        results = [1 if t['result'] == 'win' else 0 for t in self.data[symbol][-10:]]
        return sum(results) / len(results)

# Example usage (to be integrated into main pipeline):
if __name__ == "__main__":
    mem = TradeMemory()
    mem.record_trade("RELIANCE.NS", "BUY CALL", "win")
    mem.record_trade("RELIANCE.NS", "BUY PUT", "loss")
    print("Win Rate:", mem.get_win_rate("RELIANCE.NS"))
