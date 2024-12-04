
import os
from os_check import *
from pybit.unified_trading import HTTP
import time
import pandas as pd
from bybit_order import buyOrder, sellOrder
from bybit_close_position import closePosition_reBuy, closePosition_reSell
from bybit_position_get import getPosition
from bybit_check_openOrders import checkOpenOrders
import logging
from logging.handlers import RotatingFileHandler
from line_notify import LineNotify

# ログ設定
log_handler = RotatingFileHandler(
    'order_point_rsi.log', 
    maxBytes=5*1024*1024,  # 5MB
    backupCount=5  # バックアップファイルの数
)
log_handler.setFormatter(logging.Formatter(
    '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S %z'
))
logging.getLogger().addHandler(log_handler)
logging.getLogger().setLevel(logging.INFO)

# # デバッグ用のログを追加
# logger.info("Logger is set up correctly")

line_notify = LineNotify()


def order_point(RSI9, RSI14, STOCH_SLOWK, STOCH_SLOWD):
    # データを受け取って発注処理を行う
    # 発注処理の実装
    try:
        logging.info("Starting search Order_Points")
        
        # 発注トリガー判定
        is_buy_trigger = STOCH_SLOWK.iloc[-1] > STOCH_SLOWD.iloc[-1] and RSI9.iloc[-1] > RSI14.iloc[-1] and STOCH_SLOWD.iloc[-1] < 30  # RSI9.iloc[-1] < 35
        is_sell_trigger = STOCH_SLOWK.iloc[-1] < STOCH_SLOWD.iloc[-1] and RSI9.iloc[-1] < RSI14.iloc[-1] and STOCH_SLOWD.iloc[-1] > 80  # RSI9.iloc[-1] > 70
 
        # 最新のトリガー
        last_buy_triger = is_buy_trigger
        last_sell_triger = is_sell_trigger

        # 決済トリガ
        close_buy_posi = RSI9.iloc[-1] > 70
        close_sell_posi = RSI9.iloc[-1] < 30
        
        session = HTTP(testnet=True, api_key= os.getenv("bybit_apiKey"), api_secret= os.getenv("bybit_secret"),)
        tickers = session.get_tickers(category="linear", symbol="BTCUSDT",)

        # 判断結果の返却
        if last_buy_triger :
            position = getPosition()    # 建玉を持っているか確認
            logging.info(f"Position: {position}")
            orders = checkOpenOrders()  # オーダーが残っていればキャンセルする
            logging.info("=== Cansel All Orders ! ===")

            if position == 'None':
                logging.info("=== BUY ORDER ===")
                line_notify.send("=== BUY ORDER ===")
                buyOrder()
                time.sleep(5)
            else:
                pass
        
        elif last_sell_triger:
            position = getPosition()    # 建玉を持っているか確認
            logging.info(f"Position: {position}")
            orders = checkOpenOrders()  # オーダーが残っていればキャンセルする
            logging.info("=== Cansel All Orders ! ===")
            
            if position == 'None':
                logging.info("=== SELL ORDER ===")
                line_notify.send("=== SELL ORDER ===")
                sellOrder()
                time.sleep(5)
            else:
                pass
        
        elif close_buy_posi:
            position = getPosition()    # 建玉を持っているか確認
            logging.info(f"Position: {position}")
            orders = checkOpenOrders()  # オーダーが残っていればキャンセルする
            logging.info("=== Cansel All Orders ! ===")
            
            if position == 'Buy':
                logging.info("== BuyPosi_Close ===")
                line_notify.send("== BuyPosi_Close ==")
                closePosition_reBuy()
            else:
                pass
        
        elif close_sell_posi:
            position = getPosition()    # 建玉を持っているか確認
            logging.info(f"Position: {position}")
            orders = checkOpenOrders()  # オーダーが残っていればキャンセルする
            logging.info("=== Cansel All Orders ! ===")

            if position == 'Sell':
                logging.info("=== SellPosi_Close ===")
                line_notify.send("== SellPosi_Close ==")
                closePosition_reSell()
            else:
                pass
        
        else:
            pass
    except Exception as e:
        logging.error(f'An error occurred in order_point.py: {e}')
        line_notify.send(f'An error occurred in order_point.py: {e}')
        print(f'An error occurred in oorder_point.py: {e}')

