import os
import json
import importlib
import pandas as pd
from finrock.state import State
from finrock.indicators import Indicator

"""
このクラスは、Pandas DataFrameを使用して金融データを提供し、
    指定されたインジケーターを適用するためのメソッドを提供します。
"""
class PdDataFeeder:
    def __init__(
            self, 
            df: pd.DataFrame,  # データフレーム（金融データ）
            indicators: list = [],  # 使用するインジケータのリスト
            min: float = None,  # 最小値（オプション）
            max: float = None   # 最大値（オプション）
            ) -> None:
        # インスタンス変数の初期化
        self._df = df   # データフレームをインスタンス変数に保存
        self._min = min  # 最小値をインスタンス変数に保存
        self._max = max  # 最大値をインスタンス変数に保存
        self._indicators = indicators  # インジケータをインスタンス変数に保存
        self._cache = {}  # キャッシュ用の辞書を初期化

        # データフレームの検証
        assert isinstance(self._df, pd.DataFrame) == True, "df must be a pandas.DataFrame"
        assert 'timestamp' in self._df.columns, "df must have 'timestamp' column"
        assert 'open' in self._df.columns, "df must have 'open' column"
        assert 'high' in self._df.columns, "df must have 'high' column"
        assert 'low' in self._df.columns, "df must have 'low' column"
        assert 'close' in self._df.columns, "df must have 'close' column"

        # インジケータの検証
        assert isinstance(self._indicators, list) == True, "indicators must be an iterable"  # インジケータがリストにあることを確認 
        assert all(isinstance(indicator, Indicator) for indicator in self._indicators) == True, "indicators must be a list of Indicator objects"  # 各インジケータがIndicatorのインスタンスであることを確認


    @property
    def __name__(self) -> str:  # クラス名を取得するプロパティ
        return self.__class__.__name__  # クラスの名前を文字列として返す

    @property
    def name(self) -> str:  # (外部から)インジケータの名前を(簡単に)取得するためにプロパティ
        return self.__name__  # __name__プロパティを呼び出してクラス名を返す

    @property
    def min(self) -> float:   # インジケータの最小値を取得するプロパティ
        # self._minが設定されている場合、その値を返す。設定されていない場合は、データフレームのlowカラムの最小値を計算して返す
        return self._min or self._df['low'].min()
    
    @property
    def max(self) -> float:   # インジケータの最大値を取得するプロパティ
        # self._maxが設定されている場合、その値を返す。設定されていない場合は、データフレームのhighカラムの最大値を計算して返す
        return self._max or self._df['high'].max()

    def __len__(self) -> int:   # データフレームの行数を返すメソッド(データサイズの確認用)
        return len(self._df)
    
    # インデックスを指定してデータを取得するメソッド
    def __getitem__(self, idx: int, args=None) -> State:
        # idx: インデックス(整数)で、データフレームから特定の行を取得,args: その他の引数（未使用）  
        
        # キャッシュ機能を使用してトレーニングを高速化
        if idx in self._cache:
            return self._cache[idx]  # キャッシュにインデックスが存在する場合、その結果を返す

        indicators = []  # インジケータの結果を格納するリストを初期化

        # _indicatorsリストに保存された各インジケータに対し、指定されたインデックスを渡して結果を取得
        for indicator in self._indicators:
            results = indicator(idx)
            if results is None:  # インジケータがNoneを返した場合
                self._cache[idx] = None  # そのインデックスの結果をキャッシュにNoneとして保存し処理終了
                return None
            
            indicators.append(results)  # インジケータの結果をリストに追加

        # データフレームから指定されたインデックスの行を取得 
        data = self._df.iloc[idx]

        # Stateオブジェクトを作成
        state = State(
            timestamp=data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data.get('volume', 0.0),  # 取引量(存在しない場合は0.0をデフォルト)
            indicators=indicators            # インジケータの結果を渡す
        )
        self._cache[idx] = state   # キャッシュに作成したStateオブジェクトを保存(再計算を避けるため)

        return state   # 作成したStateオブジェクトを返す
    
    # クラスのインスタンスを反復可能にするために実装されるメソッド
    def __iter__(self) -> State: # type: ignore
        """ Create a generator that iterate over the Sequence."""
        # DFの行数に基づいてインデックスをループし、各インデックスに対してStateオブジェクトを生成して返す
        for index in range(len(self)):  
            yield self[index]

    # インジケーターの設定をJSONファイルに保存するメソッド
    def save_config(self, path: str) -> None:
        config = {
            "indicators": [],   # インジケータの設定を格納するリスト
            "min": self.min,    # 最小値を設定
            "max": self.max     # 最大値を設定
        }
        for indicator in self._indicators:  # 各インジケータの設定を取得してリストに追加
            config["indicators"].append(indicator.config())

        # 設定をJSONファイルに保存
        with open(os.path.join(path, "PdDataFeeder.json"), 'w') as outfile:
            # 指定されたパスにファイルを作成し、設定をJSON形式で書き込む
            json.dump(config, outfile, indent=4)  # JSONデータを整形して書き込む

     # JSONファイルから設定を読み込み、PdDataFeederインスタンスを作成する静的メソッド
    @staticmethod
    def load_config(df, path: str) -> None:
        #　設定ファイルのパスを作成
        config_path = os.path.join(path, "PdDataFeeder.json")
        # 設定ファイルが存在しない場合
        if not os.path.exists(config_path):
            raise Exception(f"PdDataFeeder Config file not found in {path}")  # 例外を発生させる
        
        with open(config_path) as json_file:  # 設定ファイルを開き、JSONデータを読み込む
            config = json.load(json_file)   # JSON形式の設定を辞書として取得

        _indicators = []   # インジケータのリストを初期化
        # 設定された各インジケータに対する処理
        for indicator in config["indicators"]:
            # インジケータ名を使用して、適切なインジケータクラスをインポート
            indicator_class = getattr(importlib.import_module(".indicators", package=__package__), indicator["name"])
            # インジケータクラスのインスタンスを作成
            ind = indicator_class(data=df, **indicator)   # DataFrameとその他の設定を渡す
            _indicators.append(ind)  # 作成したインジケータをリストに追加

        # PdDataFeederインスタンスを作成し、設定に基づいて初期化
        pdDataFeeder = PdDataFeeder(df=df, indicators=_indicators, min=config["min"], max=config["max"])

        return pdDataFeeder  # 作成したPdDataFeederインスタンスを返す