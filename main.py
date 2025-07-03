from data.fetch_data import get_intraday_data
from models.model_xgb import FNOSmartAI
from simulator.paper_trader import PaperTrader

symbols = ["^NSEI", "RELIANCE.NS", "TCS.NS"]
best_signal = None
best_accuracy = 0
best_model = None
best_df = None
best_symbol = ""

for symbol in symbols:
    try:
        df = get_intraday_data(symbol)
        if df.empty or len(df) < 30:
            print(f"Skipping {symbol}: not enough data")
            continue
        model = FNOSmartAI()
        acc = model.train(df)
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_df = df
            best_symbol = symbol
    except Exception as e:
        print(f"Error loading or training {symbol}: {e}")


signal = best_model.predict(best_df)
trader = PaperTrader()
price = float(best_df['Close'].iloc[-1])
trade_type = "BUY CALL" if signal == 1 else "BUY PUT"
trader.place_trade(trade_type, price, best_symbol)

print("Best Symbol:", best_symbol)
print("Model Accuracy:", best_accuracy)
print("Trade Signal:", trade_type)
print("Trade Log:", trader.get_trade_log())