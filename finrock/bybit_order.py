# https://bybit-exchange.github.io/docs/v5/order/create-order

import os
from os_check import *
from pybit.unified_trading import HTTP
from pprint import pprint
import logging

logging.basicConfig(
    filename='predict_result.log', 
    level=logging.INFO, 
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
    )

try:
    session = HTTP(
        testnet=True,
        api_key= os.getenv("bybit_apiKey"),
        api_secret= os.getenv("bybit_secret"),
    )

    tickers = session.get_tickers(category="linear", symbol="BTCUSDT",)

    buyOrderPrice = round(float(tickers["result"]["list"][0]["bid1Price"]) + 0.05, 2)
    sellOrderPrice = round(float(tickers["result"]["list"][0]["ask1Price"]) - 0.05, 2)

    def buyOrder():
        try:
            data = session.place_order(
                category="linear",
                symbol="BTCUSDT",
                side="Buy",
                orderType="Market",     # Market:成行、Limit:指値
                qty="0.01",
                price = buyOrderPrice,
                timeInForce="PostOnly",
            )
            logging.info("---------order Buy---------")
            logging.info(f'Buy_orderId: {data["result"]["orderId"]}')
        except Exception as e:
            logging.error(f'ERROR Order BUY: {e}')  

    def sellOrder():
        try:
            data = session.place_order(
                category="linear",
                symbol="BTCUSDT",
                side="Sell",
                orderType="Market",     # Market:成行
                qty="0.01",
                price = sellOrderPrice,
                timeInForce="PostOnly",
            )
            logging.info("---------order SELL---------")
            logging.info(f'Sell_orderId: {data["result"]["orderId"]}')
        except Exception as e:
            logging.error(f'ERROR Oder SELL: {e}')


except Exception as e:
    logging.error(f'session error(order): {e}')