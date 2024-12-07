import os
import json
import importlib
import pandas as pd
import logging
import typing
from my_state import State

"""
このクラスは、Pandas DataFrameを使用して金融データを提供し、
    指定されたインジケーターを適用するためのメソッドを提供します。
"""
class PdDataFeeder:
    def __init__(self, df: pd.DataFrame = None,  # データフレーム（金融データ）
                 output_transformer: typing.Callable = None, 
            min: float = None,  # 最小値（オプション）
            max: float = None   # 最大値（オプション）
            ) -> None:

        # ロガーを設定
        self.logger = logging.getLogger('data_feeder_logger')
        self.logger.setLevel(logging.INFO)  # ログレベルを設定

        # ファイルハンドラを作成
        file_handler = logging.FileHandler("data_feeder.log")
        file_handler.setLevel(logging.INFO)  # ファイルハンドラのログレベルを設定

        # ログフォーマットを設定
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # ロガーにファイルハンドラを追加
        self.logger.addHandler(file_handler)


        # データフレームの検証
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        if df.isna().any().any():
            raise ValueError("Input data contains NaN values.")
        if df.empty:
            raise ValueError("Input data is empty.")

        # 必要なカラムの存在を確認
        required_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'SMA7', 'RSI14',
            'MACD', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_middle', 'BB_lower',
            'PSAR',
            ]
        
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"DataFrame must have '{column}' column.")

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame is missing the following required columns: {', '.join(missing_columns)}")

        # インスタンス変数の初期化
        self._df = df.reset_index(drop=True)   # データフレームをインスタンス変数に保存(インデックスをリセット)
        
        self._min = min  # 最小値をインスタンス変数に保存
        self._max = max  # 最大値をインスタンス変数に保存
        self._output_transformer = output_transformer  # output_transformerをインスタンス変数に保存
        self._cache = {}  # キャッシュ用の辞書を初期化

        # 初期化完了のログ
        self.logger.info("PdDataFeeder initialized successfully.")        

    @property
    def data(self) -> pd.DataFrame:
        """ データフレームを返すプロパティ """
        return self._df

    def get_data(self) -> pd.DataFrame:
        """ データフレームを返すメソッド """
        return self.data  # dataプロパティを利用してデータフレームを返す

    def update_data(self, df: pd.DataFrame):

        # データフレームの検証
        required_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'SMA7','RSI14',
            'MACD', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_middle', 'BB_lower',
            'PSAR',
            ]
        
        # 必要なカラムが全て存在するか確認
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"DataFrame must have '{column}' column.")

        # カラムのデータ型チェック
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            raise ValueError("The 'timestamp' column must be of datetime type.")

        for column in ['open', 'high', 'low', 'close', 'volume']:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"The '{column}' column must be numeric.")

        # NaN値のチェック
        if df.isna().any().any():
            raise ValueError("Input data contains NaN values.")

        # 値の範囲チェック（必要に応じて）
        if (df['open'] < 0).any() or (df['high'] < 0).any() or (df['low'] < 0).any() or (df['close'] < 0).any():
            raise ValueError("Price columns must not contain negative values.")

        if (df['volume'] < 0).any():
            raise ValueError("Volume must not contain negative values.")

         # タイムスタンプの重複確認
        if df['timestamp'].duplicated().any():
            raise ValueError("The 'timestamp' column must not contain duplicate values.")
        
        # 既存のデータと新しいデータの重複を確認
        if not self._df.empty:
            if df['timestamp'].isin(self._df['timestamp']).any():
                raise ValueError("New data contain duplicate timestamps with existing data.")

        df = df.reset_index(drop=True)   # データフレームのインデックスをリセット

        # データフレームをインスタンス変数に保存
        self._df = pd.concat([self._df, df]).drop_duplicates(subset='timestamp').reset_index(drop=True)
        self.logger.info(f"Data updated in PdDataFeeder: {self._df.head()}")  # データの先頭をログに記録
        print(f"Data updated in PdDataFeeder: {self._df.head()}")   # デバック用に表示


    # @property
    # def __name__(self) -> str:  # クラス名を取得するプロパティ
    #     return self.__class__.__name__  # クラスの名前を文字列として返す

    @property
    def name(self) -> str:  # (���部から)インジケータの名前を(簡単に)取得するためにプロパティ
        return self.__class__.__name__  # __name__プロパティを呼び出してクラス名を返す

    @property
    def min(self) -> float:   # インジケータの最小値を取得するプロパティ
        # self._minが設定されている場合、その値を返す。設定されていない場合は、データフレームのlowカラムの最小値を計算して返す
        return self._min or self._df['low'].min() if not self._df.empty else None
    
    @property
    def max(self) -> float:   # インジケータの最大値を取得するプロパティ
        # self._maxが設定されている場合、その値を返す。設定されていない場合は、データフレームのhighカラムの最大値を計算して返す
        return self._max or self._df['high'].max() if not self._df.empty else None

    def __len__(self) -> int:   # データフレームの行数を返すメソッド(データサイズの確認用)
        return len(self._df)
    
    # インデックスを指定してデータを取得するメソッド
    def __getitem__(self, idx: int, args=None) -> State:
        # idx: インデックス(整数)で、データフレームから特定の行を取得,args: その他の引数（未使用）  
        
        # キャッシュ機能を使用してトレーニングを高速化
        if idx in self._cache:
            return self._cache[idx]  # キャッシュにインデックスが存在する場合、その結果を返す

        # データフレームから指定されたインデックスの行を取得 
        data = self._df.iloc[idx]

        # デバッグ情報を追加
        print(f"Index: {idx}, Data: {data}")  # 取得したデータを表示

        # データがSeriesであることを確認
        if not isinstance(data, pd.Series):
            raise TypeError(f"Expected data to be a pandas Series, got {type(data)} instead.")

        # Stateオブジェクトを作成
        try:
            state = State(
                timestamp=pd.Timestamp(data['timestamp']), 
                open=float(data['open']),
                high=float(data['high']),
                low=float(data['low']),
                close=float(data['close']),
                volume=float(data['volume']), 
                indicators=data.drop(['timestamp', 'open', 'high', 'low', 'close', 'volume']).to_dict()  # インジケータの結果を渡す
                )
        except KeyError as e:
            raise KeyError(f"Missing Key in data: {e}")
        
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
            "min": self.min,    # 最小値を設定
            "max": self.max     # 最大値を設定
        }

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

        pdDataFeeder = PdDataFeeder(df=df, min=config.get("min"), max=config.get("max"))

        return pdDataFeeder  # 作成したPdDataFeederインスタンスを返す
    


        # インジケータの計算
        # for indicator in self._indicators:
        #     try:
        #         indicator.compute()
        #         self.logger.debug(f"Calculated {indicator.__class__.__name__}: {indicator.values}")
        #     except Exception as e:
        #         self.logger.error(f"Error calculation {indicator.__class__.__name__}: {e}")

        # if not self._indicators:   # リストが空でないことを確認
        #     print('Indicators list is empty.')
            
        # self.logger.info(f"Data updated in PdDataFeeder: {self._df.head()}")
        # # データフレームの検証
        # assert isinstance(self._df, pd.DataFrame), "df must be a pandas.DataFrame"
        # assert 'timestamp' in self._df.columns, "df must have 'timestamp' column"
        # assert 'open' in self._df.columns, "df must have 'open' column"
        # assert 'high' in self._df.columns, "df must have 'high' column"
        # assert 'low' in self._df.columns, "df must have 'low' column"
        # assert 'close' in self._df.columns, "df must have 'close' column"
        # assert 'volume' in self._df.columns, "df must have 'volume' column"

        # # インジケータの検証
        # assert isinstance(self._indicators, list), "indicators must be an iterable"  # インジケータがリストにあることを確認 
        # assert all(isinstance(indicator, Indicator) for indicator in self._indicators), "indicators must be a list of Indicator objects"  # 各インジケータがIndicatorのインスタンスであることを確認


        # indicators = []  # インジケータの結果を格納するリストを初期化

        # # _indicatorsリストに保存された各インジケータに対し、指定されたインデックスを渡して結果を取得
        # for indicator in self._indicators:
        #     results = indicator(idx)
        #     if results is None:  # インジケータがNone���返した場合
        #         self._cache[idx] = None  # そのインデックスの結果をキャッシュにNoneとして保存し処理終了
        #         return None
            
        #     indicators.append(results)  # インジケータの結果をリストに追加
