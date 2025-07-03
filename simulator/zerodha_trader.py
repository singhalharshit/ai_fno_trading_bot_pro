# simulator/zerodha_trader.py

from kiteconnect import KiteConnect, KiteTicker
import time

class ZerodhaTrader:
    def __init__(self, api_key, access_token):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self.order_id = None
        self.log = []

    def place_trade(self, trade_type, price, symbol):
        try:
            order_type = self.kite.ORDER_TYPE_MARKET
            transaction_type = self.kite.TRANSACTION_TYPE_BUY if "CALL" in trade_type else self.kite.TRANSACTION_TYPE_SELL

            self.order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=1,
                order_type=order_type,
                product=self.kite.PRODUCT_MIS
            )

            self.log.append({
                "symbol": symbol,
                "type": trade_type,
                "status": "ORDER PLACED",
                "order_id": self.order_id
            })
            print(f"✅ Order placed for {symbol}: {trade_type}")
        except Exception as e:
            self.log.append({"symbol": symbol, "type": trade_type, "status": "FAILED", "error": str(e)})
            print("❌ Order placement failed:", e)

    def evaluate_trade(self, symbol, initial_price, wait_secs=600):
        try:
            print(f"⏳ Waiting {wait_secs//60} minutes to evaluate PnL...")
            time.sleep(wait_secs)

            quote = self.kite.ltp(f"NSE:{symbol}")
            new_price = quote[f"NSE:{symbol}"]["last_price"]

            pnl = new_price - initial_price if new_price and initial_price else 0
            print(f"💹 PnL: {pnl:.2f} | {initial_price} ➡ {new_price}")

            return "win" if pnl > 0 else "loss"
        except Exception as e:
            print("❌ Error fetching LTP:", e)
            return "loss"

    def get_trade_log(self):
        return self.log
