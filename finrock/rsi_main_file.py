# main file

import asyncio
import os
import pybotters
import numpy as np
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pytz import timezone
import talib
import multiprocessing as mp
from os_check import *
from line_notify import LineNotify

line_notify = LineNotify()
# ログ設定
log_handler = RotatingFileHandler(
    'main_file_rsi.log', 
    maxBytes=5*1024*1024,  # 5MB
    backupCount=5  # バックアップファイルの数
)
log_handler.setFormatter(logging.Formatter(
    '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S %z'
))
logging.getLogger().addHandler(log_handler)
logging.getLogger().setLevel(logging.INFO)

try:
    from rich import print
except ImportError:
    pass

async def main(queue):
    try:
        logging.info("Starting main function")
        symbol = "BTCUSDT"

        apis = {
            "bybit_testnet": [os.getenv("bybit_apiKey"), os.getenv("bybit_secret")]
        }
        base_url = "https://api-testnet.bybit.com"
        async with pybotters.Client(apis=apis, base_url=base_url) as client:
            store = pybotters.BybitDataStore()
        
            await client.ws_connect(
                "wss://stream-testnet.bybit.com/v5/public/linear",
                send_json={
                    "op": "subscribe", 
                    "args": ["kline.5.BTCUSDT"]  # 分足
                    },
                hdlr_json=store.onmessage,
            )
            data = pd.DataFrame()    # 空のDataFrameを初期化
            logging.info("WebSocket connected and DataFrame initialized")

            # データを一時的に保管するためのリスト
            temp_data = []
            
            while True:
                kline = store.kline.find()
                if not kline:
                    logging.info("No new kline data recived.")
                    await store.wait()
                    continue

                logging.info(f"Received kline data: {kline}")
                # print(kline)    # check用
                # DataFrameのサイズ管理（行数がmax_rowsを超えた場合、最も古いデータを削除）
                max_rows = 13000
                if len(data) > max_rows:
                    data = data.iloc[-max_rows:]

                for entry in kline:
                    jst = timezone('Asia/Tokyo')
                    temp_data.append({
                        'timestamp': datetime.fromtimestamp(entry['timestamp']/1000, jst),
                        'open' : float(entry['open']),
                        'high' : float(entry['high']),
                        'low' : float(entry['low']),
                        'close' : float(entry['close']),
                        'volume' : float(entry['volume'])
                    })
                    # temp_date = temp_date.dropna(axis=1, how='all')  # 空のカラムを除外

                    # data = pd.concat([data, temp_date], ignore_index=True)

                # 5分ごとにデータを集約してキューに送信
                current_time = datetime.now(tz=jst)
                data_resampled = None   # 初期化

                if temp_data and (current_time - temp_data[0]['timestamp']).total_seconds() >= 300:  # 5分経過
                    # DataFrameに変換
                    df = pd.DataFrame(temp_data)
                    
                    # OHLCVデータの集約
                    data_resampled = df.resample('5min', on='timestamp').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna().reset_index()
                """
                #　5分ごとにデータを集計
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
                data_resampled = data.resample('5min').last().dropna()
                # data_resampled.index = data.index.strftime('%Y-%m-%d %H:%M:%S')
            
                # インデックスをリセットして、timestampを列として保持
                data_resampled.reset_index(inplace=True)
                # data_resampled['timestamp'] = data_resampled['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                """
                if data_resampled is not None:
                    if not data_resampled.empty:

                        # OHLCVデータをNumpy配列に変換
                        close_prices = data_resampled['close'].to_numpy()
                        high_prices = data_resampled['high'].to_numpy()
                        low_prices = data_resampled['low'].to_numpy()
                        open_prices = data_resampled['open'].to_numpy()
                        volume = data_resampled['volume'].to_numpy()
                        
                        # テクニカル指標を計算
                        data_resampled['SMA7'] = talib.SMA(close_prices, timeperiod=5)
                        # data_resampled['SMA15'] = talib.SMA(close_prices, timeperiod=15)
                        # data_resampled['SMA20'] = talib.SMA(close_prices, timeperiod=20)
                        # data_resampled['RSI9'] = talib.RSI(close_prices, timeperiod=9)
                        data_resampled['RSI14'] = talib.RSI(close_prices, timeperiod=14)

                        macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
                        data_resampled['MACD'] = macd
                        data_resampled['MACD_signal'] = macd_signal
                        data_resampled['MACD_hist'] = macd_hist

                        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                        data_resampled['BB_upper'] = bb_upper
                        data_resampled['BB_middle'] = bb_middle
                        data_resampled['BB_lower'] = bb_lower

                        data_resampled['PSAR'] = talib.SAR(data_resampled['high'], data_resampled['low'], acceleration=0.02, maximum=0.2)

                        # data_resampled['ADX'] = talib.ADX(data_resampled['high'], data_resampled['low'], close_prices, timeperiod=14)
                        # data_resampled['BOP'] = talib.BOP(open_prices, high_prices, low_prices, close_prices)

                        # stoch_k, stoch_d = talib.STOCH(data_resampled['high'], data_resampled['low'], close_prices, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
                        # data_resampled['STOCH_SLOWK'] = stoch_k
                        # data_resampled['STOCH_SLOWD'] = stoch_d

                        # 各列の長さを確認
                        logging.debug(f"Data resampled shape: {data_resampled.shape}")
              
                     
                        await write_to_log(data_resampled.round(4))   # ログファイルに書き込む
                        logging.info(f"Data resampled and written to log: {data_resampled.round(4).tail(10)}")
                        
                    # # CSVの読み込み
                    # log_data = pd.read_csv(log_file_path)

                    # # 必要な指標が全て計算されている行のみをフィルタリング
                    # required_indicators = ['MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower', 'PSAR']
                    # filtered_data = log_data.dropna(subset=required_indicators)

                        # フィルタリングされたデータをキューに送信
                        for index, row in data_resampled.iterrows():
                            if not row.empty and not row.isna().all():  # rowが空でないか、すべてがNAでないか確認
                                # timestampを文字列に変換
                                timestamp = row['timestamp']
                                if isinstance(timestamp, pd.Timestamp):
                                    timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')  # 文字列に変換
                                row['timestamp'] = timestamp  # 変換したtimestampを再設定
                                
                                row_df = row.to_frame().T  # SeriesをDataFrameに変換
                                # row_df = row_df.drop('Unnamed: 0', axis=1)
                                if not queue.empty():
                                    existing_data = queue.get()
                                    if existing_data['timestamp'].iloc[0] == row['timestamp']:
                                        logging.warning(f"Duplicate timestamp detected:{row['timestamp']}.Skipping this entry.")
                                        continue
                                if not row_df.empty and not row_df.isna().any().any():  # NaNがないか確認
                                    queue.put(row_df)  
                                    logging.info(f"Data row put into Queue: {row_df}")
                                else:
                                    logging.warning(f"Row is empty after conversion as index {index}")
                            else:
                                logging.warning(f"Skipping empty or all-NA row at index {index}")

                        logging.info(f"Queue size after put: {queue.qsize()}")   # デバック用
                    
                temp_data = []  # 一時データをクリア
                await store.wait()
    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except TypeError as e:
        logging.error(f"TypeError: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        # indicators = []   # インジケータリストの初期化
         
# ログファイルのパスを定義
log_file_path = 'data_log_5m.csv'

async def write_to_log(data_resampled):
    try:
        # ログファイルが存在し、かつ空でないことを確認
        if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
            log_data = pd.read_csv(log_file_path, index_col='timestamp', parse_dates=True)  
            if 'Unnamed: 0' in log_data.columns:
                log_data = log_data.drop(columns=['Unnamed: 0'])
        else:
            # ファイルが存在しない、または空の場合、data_resampledと同じ構造の空のDataFrameを作成
            log_data = pd.DataFrame(columns=data_resampled.columns)
            if 'Unnamed: 0' in log_data.columns:
                log_data = log_data.drop(columns=['Unnamed: 0'])

        # 新しいデータでログデータを更新
        for index, row in data_resampled.iterrows():
            timestamp = row['timestamp']
            # timestampを文字列に変換
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')  # 文字列に変換
            # 行をlog_dataに追加する前に、全てがNaNでないことを確認
            if not row.isnull().all():  # 行が全てNaNでない場合にのみ追加
                log_data.loc[timestamp] = row
            else:
                logging.warning(f"Row at index {index} is all NaN and will be skipped.")
            
        # ログファイルに書き込み
        log_data.to_csv(log_file_path, index=True)
    except Exception as e:
        logging.error(f'An error occurred while writing to log: {e}')
        print(f'AN error occurred while writing to log: {e}')

if __name__ == "__main__":
    try:
        queue = mp.Queue()
        asyncio.run(main(queue))
        
    except KeyboardInterrupt:
        pass



# # (前回のデータを残さずに新たに上書きして記録する場合)
# async def write_to_log(data_resampled):
#     try:
#         # ログファイルが存在し、かつ空でないことを確認
#         if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
#             # log_data = pd.read_csv(log_file_path, index_col='time')
#             log_data = pd.read_csv(log_file_path)
#         else:
#             # ファイルが存在しない、または空の場合、data_resampledと同じ構造の空のDataFrameを作成
#             log_data = pd.DataFrame(columns=data_resampled.columns)

#         # 新しいデータでログデータを更新
#         for index, row in data_resampled.iterrows():
#             if not row.empty and not row.isna().all():  # rowが空でないかすべてがNAでないかを確認
#                 row = row.dropna(axis=1, how='all')  # すべてがNAのカラムを削除
#                 log_data.loc[index] = row    # その後、データを追加
#             else:
#                 # Handle the case where row is empty or all NA
#                 # For example, you might want to log a warning or skip the assignment
#                 logging.warning(f"Skipping empty or all-NA row at index {index}")

#         # ログファイルに書き込み
#         log_data.to_csv(log_file_path)   # 上書きモード
#         # log_data.to_csv(log_file_path, mode='a', header=not os.path.exists(log_file_path))  # 追加モード

#         # CSVの読み込み
#         log_data = pd.read_csv(log_file_path)

#         # 必要な指標が全て計算されている行のみをフィルタリング
#         required_indicators = ['MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower', 'PSAR']
#         filtered_data = log_data.dropna(subset=required_indicators)

#         # フィルタリングされたデータをキューに送信
#         for index, row in data_resampled.iterrows():
#             queue.put(row.to_frame().T)  # SeriesをdataFrameに変換しQueueに渡す
#             logging.info(f"Data row put into Queue: {row}")
            
#         print(f"Queue size after put: {queue.qsize()}")   # デバック用

#     except ValueError as e:
#         logging.error(f"ValueError: {e}")
#     except TypeError as e:
#         logging.error(f"TypeError: {e}")
#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")
#         # indicators = []   # インジケータリストの初期化



#     except Exception as e:
#         logging.error(f'An error occurred while writing to log: {e}')
#         line_notify.send(f'AN error occurred while writing to log: {e}')

# if __name__ == "__main__":
#     try:
#         queue = mp.Queue()
#         asyncio.run(main(queue))
        
#     except KeyboardInterrupt:
#         pass


