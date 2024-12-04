import os
import json
import typing
import importlib  # モジュールのインポートを動的に行うため
import numpy as np
import pandas as pd
import datetime
import logging
# from logging.handlers import RotatingFileHandler
from enum import Enum  # 列挙型を作成するため
from my_state import State, Observations  # 状態管理用のクラス
from my_data_feeder import PdDataFeeder  # データ供給者クラス
from my_reward import SimpleReward  # 報酬計算用のクラス
# from line_notify import LineNotify  # Lineメッセージ送信用

# line_notify = LineNotify()

# ログ設定
# log_handler = RotatingFileHandler(
#     'trading_env.log', 
#     maxBytes=5*1024*1024,  # 5MB
#     backupCount=5  # バックアップファイルの数
# )
# log_handler.setFormatter(logging.Formatter(
#     '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', 
#     datefmt='%Y-%m-%d %H:%M:%S %z'
# ))
# logging.getLogger().addHandler(log_handler)
# logging.getLogger().setLevel(logging.INFO)

try:
    from rich import print
except ImportError:
    pass


# 行動空間を定義する列挙型
class ActionSpace(Enum):
    DISCRETE = 3    # 離散的な行動空間
    CONTINUOUS = 2  # 連続的な行動空間
"""
アクションスペースがDISCRETE（離散）とCONTINUOUS（連続）に分かれている理由は、異なるタイプの意思決定や行動選択をモデル化するためです。特徴は以下の通りです。

1. DISCRETE（離散）
定義: 離散行動空間では、選択肢が限られており、特定の行動の中から選択します。
例: トレーディング環境では、「買う」「売る」「何もしない」といった明確な選択肢が存在します。
利点: 環境が単純で、学習アルゴリズムが扱いやすい。特に強化学習においては、Q学習などの手法が効果的に使用できます。
2. CONTINUOUS（連続）
定義: 連続行動空間では、行動が連続的な値を取ることができ、無限の選択肢があります。
例: トレーディングでは、株の購入量や売却量を連続的に調整することが考えられます。
利点: より柔軟な戦略を構築でき、特に複雑な意思決定を必要とするタスクに適しています。深層強化学習などが有効です。
"""
# トレーディング環境を定義するクラス
class TradingEnv:
    def __init__(
            self,
            data_feeder: PdDataFeeder,  # データ供給者のインスタンス
            output_transformer: typing.Callable = None,  # 出力変換関数
            initial_balance: float = 1000.0,  # 初期資産
            max_episode_steps: int = None,   # 最大エピソードステップ数
            window_size: int = 50,  # ウィンドサイズ
            reward_function: typing.Callable = SimpleReward(),  # 報酬関数
            action_space: ActionSpace = ActionSpace.DISCRETE,  # 行動空間の種類
            metrics: typing.List[typing.Callable] = [],  # 評価指標のリスト
            order_fee_percent: float = 0.001  # 注文手数料の割合
        ) -> None:

        # ロガーを設定
        self.logger = logging.getLogger('trading_env_logger')
        self.logger.setLevel(logging.INFO)  # ログレベルを設定

        # ファイルハンドラを作成
        file_handler = logging.FileHandler("trading_env.log")
        file_handler.setLevel(logging.INFO)  # ファイルハンドラのログレベルを設定

        # ログフォーマットを設定
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # ロガーにファイルハンドラを追加
        self.logger.addHandler(file_handler)

        
        self.data_feeder = data_feeder
        self.max_episode_steps = max_episode_steps
        self.output_transformer = output_transformer
        self.current_timestamp = None  # 現在のtimestampを初期化

        # データの確認
        if self.data_feeder.data.empty:
            raise ValueError("DataFeeder contains no data.")
        # データサイズの確認
        data_size = len(data_feeder)
        if data_size < 1:
            raise ValueError("Not enough data in data_feeder to initialize the environment.")

        # 最初の状態を取得
        self.current_index = 0  # 現在のインデックスを初期化
        self.current_state = self.data_feeder[self.current_index]   # 最初のStateオブジェクトを取得

        logging.info(f"Current index: {self.current_index}, Current state: {self.current_state}")
        print("Current index:", self.current_index)
        print("Current state:", self.current_state)
        # Stateが正しい型であることを確認
        if not isinstance(self.current_state, State):
            raise ValueError(f"Expected current_state to be of type State, got {type(self.current_state)} instead.")


        # コンストラクタで初期設定を行う
        self._data_feeder = data_feeder  # データ供給者を保存
        self._output_transformer = output_transformer  # 出力変換関数を保存
        self._initial_balance = initial_balance  # 初期資産を保存
        self._max_episode_steps = max_episode_steps if max_episode_steps is not None else len(data_feeder)  # 最大ステップ数を設定
        self._window_size = window_size  # ウィンドウサイズを保存
        self._reward_function = reward_function  # 報酬関数を保存
        self._metrics = metrics  # 評価指標を保存
        self._order_fee_percent = order_fee_percent  # 注文手数料を保存

        self._observations = Observations(window_size=window_size)  # 観測データの初期化
        self._observation_space = np.zeros(self.reset()[0].shape)  # 観測空間を初期化
        self._action_space = action_space  # 行動空間を保存
        self.fee_ratio = 1 - self._order_fee_percent  # 手数料を考慮した比率を計算

        # 初期化完了のログ
        self.logger.info("TradingEnv initialized successfully.")

    @property  # 行動空間の値を返すプロパティ
    def action_space(self):
        return self._action_space.value  

    @property  # プライベート変数から観測空間を返すプロパティ
    def observation_space(self):  
        return self._observation_space
    """
    オブザベーションスペースは、エージェントが環境を認識し、適切な行動を選択するための重要な要素です。強化学習の成功には、この観測空間を適切に設計し、活用することが不可欠です。
    """
    # 指定されたインデックスに基づいて観測を取得するメソッド
    def _get_obs(self, index: int, balance: float=None) -> State:
        if not isinstance(index, int):
            raise ValueError(f"Index must be an integer, got {type(index)} instead.")
        
        next_state = self.data_feeder[index]  # データ供給者から次の状態を取得
        logging.info(f"Retrieved data from data_feeder:{next_state}")
        
        logging.info(f"Type of next_state: {type(next_state)}")

        if isinstance(next_state, dict):   # next_stateが辞書であれば、Stateオブジェクトにマッピングするロジックを追加
            next_state = State(**next_state)

        elif not isinstance(next_state, State):
            logging.error(f"Expected next_state to be of type State, got {type(next_state)} instead.")
            raise ValueError(f"Expected next_state to be of type State, got {type(next_state)} instead.")      
                
        if balance is not None:  # 指定されたバランスがある場合、次の状態に設定
            next_state.balance = balance

        return next_state  # 次の状態を返す
    
    # 環境が終了したかどうかをチェックするメソッド
    def _get_terminated(self):
        return False  # 現在の実装では常にFalseを返す(終了条件は別途必要)

    # 指定されたアクションを実行し、アクションの種類とサイズを返すメソッド
    def _take_action(self, action_pred: typing.Union[int, np.ndarray]) -> typing.Tuple[int, float]:
        from bybit_order import buyOrder, sellOrder
        
        # アクションが範囲内であることを確認
        # アクションがNumPy配列(連続的な選択)の場合
        if isinstance(action_pred, np.ndarray):
            # 注文サイズを(0～1)の範囲にクリップし、小数点以下2桁に丸める
            order_size = np.clip(action_pred[1], 0, 1)
            order_size = np.around(order_size, decimals=2)
            # アクションを(-1～1)の範囲から(0～3)にスケールする
            action = int((np.clip(action_pred[0], -1, 1) + 1) * 1.5) #  -1,1 を 0,3に変換
        # アクションが離散的な場合
        elif action_pred in [0, 1, 2]:
            order_size = 1.0   # 注文サイズを1.0に設定
            action = action_pred   # アクションをそのまま使用
            # アクションが許可された範囲内にあることを確認
            assert (action in list(range(self._action_space.value))) == True, f'action must be in range {self._action_space.value}, received: {action}'
        else:
            # 無効なアクションタイプの場合はエラーを発生
            raise ValueError(f'invalid action type: {type(action)}')


        # 最後の状態と次の状態を取得（直前の2つの観測を取得）
        last_state, next_state = self._observations[-2:]

        # 残高が不足している場合は、アクションをホールド(0)に変更
        if action == 2 and last_state.allocation_percentage == 1.0:
            action = 0   # 全ての資産が投資されているため、購入できない

        # 資産が不足している場合は、アクションをホールド(0)に変更
        elif action == 1 and last_state.allocation_percentage == 0.0:
            action = 0  # 資産がないため、売却できない

        # 注文サイズが0の場合は、アクションをホールド(0)に変更
        if order_size == 0:
            action = 0  # 注文がないため、アクションをホールドに設定

        # アクションが購入(2)の場合
        if action == 2: # buy
            buy_order_size = order_size  # 購入する注文サイズを設定
        
            if not isinstance(next_state, State):
               raise ValueError(f"Expected next_state to be of type State, got {type(next_state)} instead.")
           
            """ ByBit API を使用して買い注文を実行"""
            buyOrder(size=buy_order_size)
            
            # 次の状態の割り当て割合を更新
            next_state.allocation_percentage = last_state.allocation_percentage + (1 - last_state.allocation_percentage) * buy_order_size
            # 次の状態の資産を計算
            next_state.assets = last_state.assets + (last_state.balance * buy_order_size / last_state.close) * self.fee_ratio
            # 次の状態の残高を計算
            next_state.balance = last_state.balance - (last_state.balance * buy_order_size) * self.fee_ratio
            
            logging.info(f"=== BUY ORDER ===:zandaka:{next_state.balance}")
            # line_notify.send(f"=== BUY ORDER ===:zandaka:{next_state.balance}")
        # アクションが売却(1)の場合
        elif action == 1: # sell
            sell_order_size = order_size   # 売却する注文サイズを設定
        
            if not isinstance(next_state, State):
               raise ValueError(f"Expected next_state to be of type State, got {type(next_state)} instead.")
 
            """ ByBit API を使用して売り注文を実行"""
            sellOrder(size=sell_order_size)
            
            # 次の状態の割り当て割合を更新
            next_state.allocation_percentage = last_state.allocation_percentage - last_state.allocation_percentage * sell_order_size
            # 次の状態の残高を計算
            next_state.balance = last_state.balance + (last_state.assets * sell_order_size * last_state.close) * self.fee_ratio
            # 次の状態の資産を計算
            next_state.assets = last_state.assets - (last_state.assets * sell_order_size) * self.fee_ratio

            logging.info(f"=== SELL ORDER ===:zandaka:{next_state.balance}")
            # line_notify.send(f"=== SELL ORDER ===:zandaka:{next_state.balance}")
        # それ以外（ホールド）の場合
        else: # hold
            next_state.allocation_percentage = last_state. allocation_percentage  # 割り当て割合は変更なし
            next_state.assets = last_state.assets  # 資産は変更なし
            next_state.balance = last_state.balance  # 残高は変更なし

        # 次の状態の割り当て割合が1.0を超える場合はエラー
        if next_state.allocation_percentage > 1.0:
            raise ValueError(f'next_state.allocation_percentage > 1.0: {next_state.allocation_percentage}')
        # 次の状態の割り当て割合が0.0未満の場合はエラー
        elif next_state.allocation_percentage < 0.0:
            raise ValueError(f'next_state.allocation_percentage < 0.0: {next_state.allocation_percentage}')

        return action, order_size  # アクションと注文サイズを返す
    
    @property
    def metrics(self):  # メトリクスを取得するプロパティ
        return self._metrics  # プライベート変数からメトリクスを返す

    # 観測に基づいてメトリクスを更新するメソッド
    def _metricsHandler(self, observation: State):
        metrics = {}   # メトリクスを格納する辞書を初期化
        # メトリクスをループして更新
        for metric in self._metrics:
            metric.update(observation)   # 各メトリクスを観測で更新
            metrics[metric.name] = metric.result  # 結果を辞書に保存

        return metrics  # 更新されたメトリクスを返す

    # 環境のステップを進め、次の状態、報酬、終了フラグなどを返すメソッド
    def step(self, action: int) -> typing.Tuple[State, float, bool, bool, dict]:
        # 現在のtimestampを更新
        self.current_timestamp = self.data_feeder._df.iloc[self.current_step]['timestamp']
        # ステップインデックスから次のインデックスを取得
        if not self._env_step_indexes:
            raise ValueError("No more steps availabel in the environment.")

        index = self._env_step_indexes.pop(0)  # 次のインデックスを取得

        observation = self.data_feeder[index]  # PdDataFeederからStateオブジェクトを取得
        logging.info(f"Retrieved observation from data_feeder: {observation}")

        logging.info(f"Type of observation: {type(observation)}")

        if isinstance(observation, dict):
            observation = State(**observation)

        if not isinstance(observation, State):
            raise ValueError(f"Expected observation to be of type State, got {type(observation)} instead.")
             
        logging.info(f"Observation obtained: {observation}")

        # 新しい観測で観測オブジェクトを更新
        self._observations.append(observation)  # 観測をリストに追加

        # アクションを実行し、アクションと注文サイズを取得
        action, order_size = self._take_action(action)
        
        # 報酬を計算
        try:
            reward = self._reward_function(self._observations)
        except Exception as e:
            logging.error(f"Error calculating reward: {e}")
            reward = 0.0   # エラーが発生した場合は報酬を0に設定
            
        # 環境が終了したかどうかをチェック
        terminated = self._get_terminated()
        # ステップインデックスが残っているかどうかで切り捨て(truncated)を判断
        truncated = False if self._env_step_indexes else True
        info = {
            "states": [observation],  # 現在の観測を含む辞書を作成
            "metrics": self._metricsHandler(observation)  # 更新されたメトリクスを追加
            }

        # 観測を変換
        if self._output_transformer is None:
            raise ValueError("output_transformer is not set. please provide a valid output transformer.")
        
        transformed_obs = self._output_transformer.transform(self._observations)

        # 変換された観測にNaNが含まれているかチェック
        if np.isnan(transformed_obs).any():
            raise ValueError("transformed_obs contains nan values, check your data")   # NaNが含まれている場合はエラーを発生

        # 次の状態、報酬、終了フラグ、切り捨てフラグ、追加情報を返す
        return transformed_obs, reward, terminated, truncated, info

    # 環境をリセットし、初期状態を返すメソッド
    def reset(self) -> typing.Tuple[State, dict]:
        # 最初のtimestampを取得
        self.current_timestamp = self.data_feeder._df.iloc[0]['timestamp']
        # 使用可能なデータサイズを計算
        size = len(self._data_feeder) - self._max_episode_steps
        if size <= 0:
            raise ValueError("Not enough data in data_feeder to reset the environment.")
        
        # 環境の開始インデックスをランダムに選択(データが十分な場合)
        self._env_start_index = np.random.randint(0, size) if size > 0 else 0
        # ステップインデックスを生成(開始インデックスから最大ステップ数分)
        self._env_step_indexes = list(range(self._env_start_index, self._env_start_index + self._max_episode_steps))

        # 初期観測はウィンドウサイズの最初の状態
        self._observations.reset()  # 観測オブジェクトをリセット

        # 観測が満杯になるまで新しい観測を追加
        while not self._observations.full:
            # 次のインデックスから観測を取得し、初期バランスを設定
            obs = self._get_obs(self._env_step_indexes.pop(0), balance=self._initial_balance)
            if not isinstance(obs, State):
               raise ValueError(f"Expected obs to be of type State, got {type(obs)} instead.")

            if obs is None:  # 観測が取得できない場合、スキップ
                continue
            # 新しい観測で観測オブジェクトを更新
            self._observations.append(obs)

        # 状態とメトリクス情報を格納する辞書を作成
        info = {
            "states": self._observations.observations,  # 現在の観測を追加
            "metrics": {}  # メトリクスは空の辞書で初期化
            }
        
        # 各メトリクスを最後の観測でリセット
        for metric in self._metrics:
            metric.reset(self._observations.observations[-1])

        # 観測を変換
        transformed_obs = self._output_transformer.transform(self._observations)
        # 変換された観測にNaNが含まれているかチェック
        if np.isnan(transformed_obs).any():
            raise ValueError("transformed_obs contains nan values, check your data")  # NaNがあればエラーを発生
        
        # 状態と情報を返す
        return transformed_obs, info

    # 環境の描画メソッド（未実装）
    def render(self):
        raise NotImplementedError

    # 環境を閉じるメソッド
    def close(self):
        """ Close the environment
        """
        pass  # 特に何もしない（必要な場合はここにコードを追加）

    # 環境の設定を返すメソッド
    def config(self):
        
        return {
            "data_feeder": self._data_feeder.__name__,  # データ供給者のクラス名を取得
            "output_transformer": self._output_transformer.__name__ if self._output_transformer else None,  # 出力変換者のクラス名を取得
            "initial_balance": self._initial_balance,  # 初期バランスを取得
            "max_episode_steps": self._max_episode_steps,  # 最大エピソードステップ数を取得
            "window_size": self._window_size,  # ウィンドウサイズを取得
            "reward_function": self._reward_function.__name__,  # 報酬関数のクラス名を取得
            "metrics": [metric.__name__ for metric in self._metrics],  # 各メトリクスのクラス名を取得
            "order_fee_percent": self._order_fee_percent,  # 注文手数料の割合を取得
            "observation_space_shape": tuple(self.observation_space.shape),  # 観測空間の形状をタプルとして取得
            "action_space": self._action_space.name,  # 行動空間の名前を取得
        }
    
    # 環境の設定を保存するメソッド
    def save_config(self, path: str = ""):
        # 保存先のパスを構築
        output_path = os.path.join(path, "TradingEnv.json")
        with open(output_path, "w") as f:
            json.dump(self.config(), f, indent=4)  # 設定をJSON形式でファイルに保存

    
    @staticmethod   # 環境の設定を読み込むメソッド
    def load_config(data_feeder, path: str = "", **kwargs):
        
        # 設定ファイルのパスを構築
        input_path = os.path.join(path, "TradingEnv.json")
        # 設定ファイルが存在しない場合
        if not os.path.exists(input_path):
            raise Exception(f"TradingEnv Config file not found in {path}")  # 例外を発生させる

        # 設定ファイル（JSON形式）を辞書として読み込む
        with open(input_path, "r") as f:
            config = json.load(f)

        # TradingEnvのインスタンスを生成
        environment = TradingEnv(
            data_feeder = data_feeder,  # データ供給者を設定
            output_transformer = getattr(importlib.import_module(".scalers", package=__package__), config["output_transformer"])(),  # 出力変換者を動的にインポート
            initial_balance = kwargs.get("initial_balance") or config["initial_balance"],  # 初期バランスを取得
            max_episode_steps = kwargs.get("max_episode_steps") or config["max_episode_steps"],  # 最大エピソードステップ数を取得
            window_size = kwargs.get("window_size") or config["window_size"],  # ウィンドウサイズを取得
            reward_function = getattr(importlib.import_module(".reward", package=__package__), config["reward_function"])(),  # 報酬関数を動的にインポート
            action_space = ActionSpace[config["action_space"]],  # 行動空間を設定
            metrics = [getattr(importlib.import_module(".metrics", package=__package__), metric)() for metric in config["metrics"]],  # メトリクスを動的にインポートしてリストに格納
            order_fee_percent = kwargs.get("order_fee_percent") or config["order_fee_percent"]  # 注文手数料wの割合を取得
        )
        
        return environment  # 作成した環境インスタンスを返す