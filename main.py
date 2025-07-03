# main.py (Updated with Zerodha evaluate_trade integration)

from data.fetch_data import get_intraday_data
from models.model_xgb import FNOSmartAI
from simulator.paper_trader import PaperTrader
from simulator.zerodha_trader import ZerodhaTrader
from symbol_screener_ai import get_ai_screened_symbols
from rl.self_learning import TradeMemory

import getpass

mode = input("Select mode [paper/live]: ").strip().lower()

symbols, _ = get_ai_screened_symbols()
best_signal = None
best_accuracy = 0
best_model = None
best_df = None
best_symbol = ""

# Track memory
mem = TradeMemory()

for symbol in symbols:
    df = get_intraday_data(symbol)
    if df.empty:
        continue
    model = FNOSmartAI()
    acc = model.train(df)
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_df = df
        best_symbol = symbol

if best_model is None:
    print("❌ No valid model trained. Check data or retry.")
else:
    signal = best_model.predict(best_df)
    price = best_df['Close'].iloc[-1]
    trade_type = "BUY CALL" if signal == 1 else "BUY PUT"

    if mode == "live":
        api_key = input("🔑 Zerodha API Key: ").strip()
        access_token = getpass.getpass("🔐 Zerodha Access Token: ").strip()
        trader = ZerodhaTrader(api_key, access_token)
    else:
        trader = PaperTrader()

    trader.place_trade(trade_type, price, best_symbol)

    # --- Evaluation (real PnL or simulation based on mode) ---
    if mode == "live":
        outcome = trader.evaluate_trade(best_symbol, price, wait_secs=600)
    else:
        import random
        outcome = random.choice(["win", "loss"])

    mem.record_trade(best_symbol, trade_type, outcome)

    print("\n===== AI TRADE SUMMARY =====")
    print("Best Symbol:", best_symbol)
    print("Model Accuracy:", best_accuracy)
    print("Trade Signal:", trade_type)
    print("Trade Log:", trader.get_trade_log())
    print("Win Rate (last 10):", mem.get_win_rate(best_symbol))
