# zerodha_executor.py

from kiteconnect import KiteConnect, KiteTicker
import os
import json

class ZerodhaTrader:
    def __init__(self, api_key, access_token):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)

    def get_instrument_token(self, symbol):
        all_inst = self.kite.instruments("NSE")
        for inst in all_inst:
            if inst['tradingsymbol'] == symbol:
                return inst['instrument_token']
        raise Exception("Instrument not found for symbol: " + symbol)

    def place_order(self, symbol, signal, price):
        order_type = "MARKET"
        quantity = 1

        if signal == "BUY CALL":
            transaction_type = self.kite.TRANSACTION_TYPE_BUY
        else:
            transaction_type = self.kite.TRANSACTION_TYPE_SELL

        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=self.kite.EXCHANGE_NSE,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product=self.kite.PRODUCT_MIS,
                variety=self.kite.VARIETY_REGULAR
            )
            print(f"✅ Order placed: {order_id} for {symbol} [{signal}]")
            return order_id
        except Exception as e:
            print(f"❌ Order failed for {symbol}: {e}")
            return None

# Quick test (if needed)
if __name__ == "__main__":
    trader = ZerodhaTrader(api_key="b4newrekc8ral4mk", access_token="your_access_token_here")
    trader.place_order("RELIANCE", "BUY CALL", 2700)
