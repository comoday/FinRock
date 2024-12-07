import typing
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Union
# Stateオブジェクトは、金融データを表現するための基本的な構成要素であり、
# ObservationsクラスはこれらのStateオブジェクトを管理し、特定のウィンドウサイズの範囲で効率的に操作できるように設計されています。
# この連携により、金融データの分析やトレード戦略の実装が可能になります.

class State:
    def __init__(
            self, 
            timestamp: Union[str, pd.Timestamp], 
            open: float, 
            high: float, 
            low: float, 
            close: float, 
            volume: float,
            indicators: typing.Optional[dict] = None
        ):  # それぞれの項目をインスタンス変数に設定
        try:
            self.timestamp = pd.Timestamp(timestamp)
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {timestamp}")
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        # # indicatorsの値を数値に変換
        # self.indicators = {k: float(v) for k, v in indicators.items()}
        # # indicatorsの値を数値に変換
        if indicators is not None:
            if not isinstance(indicators, dict):
                raise ValueError("Indicators must be a dictionary.")
            self.indicators = {k: float(v) for k, v in indicators.items()}
        else:
            self.indicators = {}

        # self.indicators = indicators if indicators is not None else {}  # インジケータの辞書

        # try:
        #     self.date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        # except ValueError:
        #     raise ValueError(f'received invalid timestamp date format: {timestamp}, expected: YYYY-MM-DD HH:MM:SS')
        
        self._balance = 0.0 # 現金残高（プライベート変数）
        self._assets = 0.0 # 資産残高（プライベート変数）
        self._allocation_percentage = 0.0 # この状態に割り当てられた資産の割合（プライベート変数）


    def to_dict(self):
        """Stateオブジェクトを辞書形式に変換するメソッド"""
        state_dict = {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
        }
        if isinstance(self.indicators, dict):
            state_dict.update(self.indicators)  # インジケータを辞書に追加
        return state_dict

    def __repr__(self):
        """Stateオブジェクトの文字列表現を定義するメソッド"""
        return f"State(timestamp={self.timestamp}, open={self.open}, high={self.high}, low={self.low}, close={self.close}, volume={self.volume}, indicators={self.indicators})"    # タイムスタンプをdatetimeオブジェクトに変換
        
    @property
    def balance(self):
        return self._balance   # 現金残高を取得するプロパティ
    
    @balance.setter
    def balance(self, value: float):
        self._balance = value   # 現金残高を設定するセッター

    @property
    def assets(self):
        return self._assets    # 資産残高を取得するプロパティ
    
    @assets.setter
    def assets(self, value: float):
        self._assets = value   # 資産残高を設定するセッター

    @property
    def account_value(self):
        # アカウントの総価値を計算（現金残高 + 資産の評価額）
        return self.balance + self.assets * self.close

    @property
    def allocation_percentage(self):
        return self._allocation_percentage   # 割り当て割合を取得するプロパティ
    
    @allocation_percentage.setter
    def allocation_percentage(self, value: float):
        # 割り当て割合が0.0から1.0の範囲であることを確認
        assert 0.0 <= value <= 1.0, f'allocation_percentage value must be between 0.0 and 1.0, received: {value}'
        self._allocation_percentage = value   # 割り当て割合を設定するセッター
    

class Observations:
    def __init__(
            self, 
            window_size: int,  # ウィンドウサイズ（観測の最大値）
            observations: typing.Optional[typing.List[State]] = None,  # 初期観測リスト（デフォルトは空リスト）
        ):
        self._observations = observations if observations is not None else [] # 観測リストをプライベート変数に設定
        self._window_size = window_size  # ウィンドウサイズをプライベート変数に設定

        # 観測がリストであることを確認
        assert isinstance(self._observations, list), "observations must be a list"
        # 観測の長さがウィンドサイズ以下であることを確認
        assert len(self._observations) <= self._window_size, f'observations length must be <= window_size, received: {len(self._observations)}'
        # 観測リストのすべての要素がStateオブジェクトであることを確認
        assert all(isinstance(observation, State) for observation in self._observations), "observations must be a list of State objects"

    def __len__(self) -> int:
        # 観測リストの長さを返す
        return len(self._observations)
    
    @property
    def window_size(self) -> int:
        # ウィンドサイズを取得するプロパティ
        return self._window_size
    
    @property
    def observations(self) -> typing.List[State]:
        # 観測リストを取得するプロパティ
        return self._observations
    
    @property
    def full(self) -> bool:
        # 観測リストが満杯かどうかを確認するプロパティ
        return len(self._observations) == self._window_size

    def __getitem__(self, idx: int) -> State:
        # 指定されたインデックスの観測を取得
        try:
            return self._observations[idx]  # インデックスに対応する観測を返す
        except IndexError:
            # インデックスが範囲外の場合は例外を発生させる
            raise IndexError(f'index out of range: {idx}, observations length: {len(self._observations)}')
        
    def __iter__(self) -> typing.Generator[State, None, None]:
        """ シーケンスを反復するジェネレータを作成します ."""
        for index in range(len(self)):
            yield self[index]  # 各インデックスの観測を生成する

    def reset(self) -> None:
        # 観測リストを空にする
        self._observations = []
        logging.info("Observations have been reset.")
    
    def append(self, state: State) -> None:
        # stateはStateオブジェクトまたは Noneであるべき
        assert isinstance(state, State) , "state must be a State object "
        self._observations.append(state)  # 観測リストに新しい状態を追加
        logging.info(f"Appended state: {state}")

        # 観測リストがウィンドウサイズを超えた場合、最初の要素を削除
        if len(self._observations) > self._window_size:
            remove_state = self._observations.pop(0)  # 最初の要素を削除してサイズを維持する
            logging.info(f"Removed state due to window size limit: {remove_state}")

    # 観測から各状態の「それぞれの値」を取得し、NumPy配列として返すプロパティ
    @property
    def close(self) -> np.ndarray:
        # 各状態のclose属性をリスト内包表記で取得
        return np.array([state.close for state in self._observations])
    
    @property
    def high(self) -> np.ndarray:
        # 各状態のhigh属性をリスト内包表記で取得
        return np.array([state.high for state in self._observations])
    
    @property
    def low(self) -> np.ndarray:
        # 各状態のlow属性をリスト内包表記で取得
        return np.array([state.low for state in self._observations])
    
    @property
    def open(self) -> np.ndarray:
        # 各状態のopen属性をリスト内包表記で取得
        return np.array([state.open for state in self._observations])
    
    @property
    def allocation_percentage(self) -> np.ndarray:
        # 各状態のallocation_percentage属性をリスト内包表記で取得
        return np.array([state.allocation_percentage for state in self._observations])

    @property
    def volume(self) -> np.ndarray:
        # 各状態のvolume属性をリスト内包表記で取得
        return np.array([state.volume for state in self._observations])