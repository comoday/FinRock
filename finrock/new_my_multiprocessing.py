# マルチプロセス(rsi_main_file.py, order_point.py)

from multiprocessing import Process
import multiprocessing as mp
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from line_notify import LineNotify
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from my_trading_env import TradingEnv
from my_data_feeder import PdDataFeeder
from my_scalers import ZScoreScaler
from my_state import State
import queue

line_notify = LineNotify()
# ログ設定 multiprocessing.logのサイズが5MBを超えると、新しいログファイルが作成され、最大5つのバックアップファイルが保持されます。古いログファイルは自動的に削除
def setup_logger(name, log_file, level=logging.INFO):
    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S %z'
    ))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
# Queueの作成
q = mp.Queue()

# main_file.pyのmain関数を実行するためのラッパー
def run_main_file(q):
    logger = setup_logger('main_file_logger', 'multiprocessing_rsi_main.log')
    try:
        logger.info("Starting run_rsi_main_file")
        import rsi_main_file
        asyncio.run(rsi_main_file.main(q))
        logger.info("Finished run_rsi_main_file")
    except Exception as e:
        logger.error(f"Error in run_rsi_main_file: {e}")


# TradingEnvを使用して発注処理を行うためのラッパー
def run_trading_env(q, delay_minutes=170):
    logger = setup_logger('trading_env_logger', 'multiprocessing_trading_env.log')
    logger.info(f"Starting run_trading_env with a delay of {delay_minutes} minutes")

    try:
        time.sleep(delay_minutes * 60)   # delay_minutes: 待ち時間
        
        logger.info("Delay completed. Starting run_trading_env.")
        line_notify.send("Delay completed. Starting run_trading_env.")
        
              
        # キューからデータを取得する。
        data_frames = []   # 複数のデータフレームを格納するリスト
        start_time = time.time()  # ループの開始時間
        timeout_duration = 301  # タイムアウトの設定（秒）
        
        previous_data = None  # 前回のデータを保持するための変数

        while True:
            try:
                df = q.get(timeout=timeout_duration)  # キューからデータを取得
                logger.info(f"Queue size after getting data: {q.qsize()}")
                if isinstance(df, pd.DataFrame) and not df.isna().any().any():   # NaNがないか確認
                    # データが変更されたか確認
                    if previous_data is None or not df.equals(previous_data):
                        data_frames.append(df)
                        logger.info(f"Received data from queue: {df}")
                        # line_notify.send(f"Received data from queue: {q.qsize}")
                        previous_data = df   # 現在のデータを保存
                    else:
                        logger.info("Received duplicate data, skipping.")
                else:
                    logger.warning("Recived data is not a valid DataFrame or contains NaN values.")
            except queue.Empty:
                logger.warning("Queue is empty, waiting for more data...")
                continue
            except Exception as e:
                logger.error(f"Error getting data from queue: {e}")
                break   # データ取得に失敗した場合はループを終了
        
            # タイムアウトの確認
            if time.time() - start_time > timeout_duration:
                logger.warning("Timeout reached while waiting for data.")
                line_notify.send("Timeout reached while waiting for data.")
                break  # タイムアウトに達した場合、ループを終了

        # すべてのデータフレームを結合
        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)
            try:
                output_transformer = ZScoreScaler() # 適切な出力変換器を指定(この場合ZScoreScaler)
                # data_feederの初期化（適切なパラメータで初期化）
                data_feeder = PdDataFeeder(df=combined_df, output_transformer=output_transformer)  
                logger.info("Initialized PdDataFeeder successfully.")
            except ValueError as e:
                logger.error(f"Error initializing PdDataFeeder: {e}")
                return
        else:
            logger.error("No valid data received to initialize PdDataFeeder.")
            return   # 処理を中断               
    
        # PdDataFeederのデータサイズを取得
        data_size = len(data_feeder)   # データフレームの行数を取得
        logger.info(f"Data size in PdDataFeeder: {data_size}") 
        line_notify.send(f"Data size in PdDataFeeder: {data_size}")
        # max_episode_stepsをデータサイズに基づいて設定
        max_episode_steps = min(50, data_size)  # ウィンドウサイズを50に設定し、データサイズを超えないようにする
        
        logger.info(f"Max episode steps:{max_episode_steps}, Data size:{data_size}")

        if max_episode_steps < 1:
            logger.error("Not enough data to initialize TradingEnv.")
            return  # データが不足している場合は処理を中断

        logger.info("Initialized PdDataFeeder. Next TradingEnv..")
        line_notify.send("Initialized PdDataFeeder. Next TradingEnv..")

        # データを確認し、必要に応じて変換
        for index in range(len(data_feeder)):
            try:
                state = data_feeder[index]
                if isinstance(state, dict):  # もしstateが辞書であれば
                    # Stateオブジェクトに変換
                    state = State(
                        timestamp=int(pd.to_datetime(state['timestamp']).to_timestamp()),
                        open=float(state['open']),
                        high=float(state['high']),
                        low=float(state['low']),
                        close=float(state['close']),
                        volume=float(state['volume']),
                        indicators={k: float(v) for k, v in state.items() if k not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']}
                    )
                    data_feeder._df.iloc[index] = state  # 変換したStateオブジェクトをデータフレームに戻す
                elif isinstance(state, State):
                    continue
                else:
                    raise ValueError(f"Unexpected data format at index {index}: {type(state)}")
            except Exception as e:
                logger.error(f"Error processing index {index}: {e}")    
        
        # timestampをdatetimeに変換
        # data_feeder._df['timestamp'] = pd.to_datetime(data_feeder._df['timestamp'])
        # logger.info("timestamp chenged datetime succesfully.")
        
        # TradingEnvの初期化    
        try:
            if not isinstance(data_feeder, PdDataFeeder):
                logger.error("Invialid data feeder provided. Excepted instance of PdDataFeeder.")
                return
            
            # TradingEnvの初期化
            trading_env = TradingEnv(data_feeder=data_feeder, max_episode_steps=max_episode_steps, output_transformer=output_transformer)    
            logger.info("Initialized TradingEnv")
            line_notify.send("Initialized TradingEnv")
        except Exception as e:
            logger.error(f"Error initializing TradingEnv: {e}")
            return   # 処理の中断
                        
        # 学習済みモデル（アクターモデル）を読み込む
        try:
            model_path = "runs/1730862252"
            agent = tf.keras.models.load_model(f'{model_path}/ppo_sinusoid_actor.keras')
            logger.info("Model loader successfully.")
            line_notify.send("Model loader successfully.")   
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return   # モデルがロードできない場合は関数を終了

        # 環境をリセット
        observation, info = trading_env.reset()
        logger.info(f"Initial observation: {observation}")
        line_notify.send(f"Initial observation: {observation}")

        # indicators = []   # インジケータのリストを初期化
        # required_indicators = ['MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower', 'PSAR']

        while True:
            try:
                df = q.get(timeout=5)   # タイムアウトを設定
                if df is not None and not df.empty:
                    logger.info(f"Received complete data from queue: {df.head()}")
                    line_notify.send(f"Received complete data from queue: {df.head()}")
                    # データが変更されたか確認
                    if previous_data is None or not df.equals(previous_data):
                        data_feeder.update_data(df)                      
                        previous_data = df
                    else:
                        logger.info("Received duplicate data, skipping update.")
                else:
                    logger.warning("Received None or empty DataFrame from queue, skipping update.")
                    continue
            except Exception as e:
                logger.error(f'An error occurred while getting data from queue: {e}')
                continue   # エラーが発生した場合はループを続ける
                     
            # モデルによるアクションの予測
            logger.info("Predicting action using the model.")
            line_notify.send("Predicting action using the model.")
            prob = agent.predict(np.expand_dims(observation, axis=0), verbose=False)[0]
            action = np.argmax(prob)  # 確率が最も高いアクションを選択
            logger.debug(f'Predicted probabilities: {prob}, Selected action: {action}')  # デバック用

            # 環境のステップを進める
            logger.info("Stepping through the enviroment.")
            observation, reward, terminated, truncated, info = trading_env.step(action)
            logger.info(f"Step result - Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            
            # エピソードが終了した場合 →　環境をリセットする
            if terminated or truncated:
                logger.info("Episode finished, resetting environment.")
                observation, info = trading_env.reset()
                logger.debug(f'Environment reset - Initial observation:{observation}, Info:{info}')  #　デバック用

    except ValueError as e:
        line_notify.send(f"ValueError in run_trading_env: {e}")
        logging.error(f"ValueError in run_trading_env: {e}")
    except TypeError as e:
        line_notify.send(f"TypeError in run_trading_env: {e}")
        logging.error(f"TypeError in run_trading_env: {e}")
    except Exception as e:
        line_notify.send(f"UnexpectedError in run_trading_env: {e}")
        logging.error(f"UnexpectedError in run_trading_env: {e}")
   
if __name__ == "__main__":
    q = mp.Queue()
    # プロセスを作成
    process_main = mp.Process(target=run_main_file, args=(q,))
    process_trading_env = mp.Process(target=run_trading_env, args=(q,))
    
    process_main.start()
    process_trading_env.start()

    process_main.join()   # process_mainが終了するのを待つ
    process_trading_env.join()   # process_trading_envが終了するのを待つ