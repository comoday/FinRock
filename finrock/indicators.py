import pandas as pd

from .render import RenderOptions, RenderType, WindowType

""" Implemented indicators:
- SMA
- Bolinger Bands
- RSI
- PSAR
- MACD (Moving Average Convergence Divergence)

TODO:
- Commodity Channel Index (CCI), and X is the 
- Average Directional Index (ADX)
"""

    
class Indicator:
    """ 
    Base class for indicators
    このクラスは、様々なテクニカル指標を計算するための基底クラスである
    """
    def __init__(
            self, 
            data: pd.DataFrame,  # 入力データとしてpandasのDataFrameを受け取る
            target_column: str='close',  # 目標列名（デフォルトは'close'）
            render_options: dict={},  # レンダリングオプションを指定する辞書（デフォルトは空）
            min: float=None,  # 最小値（指定が無ければNone）
            max: float=None,  # 最大値（指定が無ければNone）
            **kwargs  # その他のキーワード引数を受け取る
        ) -> None:
        # コンストラクタ。インジケーターの設定を行う。
        self._data = data.copy()  # 入力データをコピーして保存
        self._target_column = target_column  # 目標列名を保存
        self._custom_render_options = render_options  # カスタムレンダリングオプションを保存
        self._render_options = render_options  # レンダリングオプションを初期化
        self._min = min  # 最小値を保存（初期値はNone）
        self._max = max  # 最大値を保存（初期値はNone）
        self.values = {}  # 計算されたインジケーターの値を格納する辞書を初期化

        # データがpandasのDataFrameであることを確認
        assert isinstance(self._data, pd.DataFrame) == True, "data must be a pandas.DataFrame"
        # 指定された目標列がデータフレームのカラムに存在するか確認
        assert self._target_column in self._data.columns, f"data must have '{self._target_column}' column"

        self.compute()  # 指標の計算を実行
        # カスタムレンダリングオプションが指定されていない場合、デフォルトのレンダリングオプションを設定
        if not self._custom_render_options:
            self._render_options = self.default_render_options() 

    @property
    def min(self):
        # 最小値を取得するプロパティ
        return self._min  # プライベート変数から最小値を返す

    @min.setter
    def min(self, min: float):
        # 最小値を設定するセッター
        self._min = self._min or min  # 既に最小値が設定されていない場合にのみ新しい値を設定
        if not self._custom_render_options:
            # カスタムレンダリングオプションが指定されていない場合、デフォルトのレンダリングオプションを設定
            self._render_options = self.default_render_options() 

    @property
    def max(self):
        # 最大値を取得するプロパティ
        return self._max  # プライベート変数から最大値を返す

    @max.setter
    def max(self, max: float):
        # 最大値を設定するセッター
        self._max = self._max or max  # 既に最大値が設定されていない場合にのみ新しい値を設定
        if not self._custom_render_options:
            # カスタムレンダリングオプションが指定されていない場合、デフォルトのレンダリングオプションを設定
            self._render_options = self.default_render_options() 

    @property
    def target_column(self):
        # 目標列名を取得するプロパティ
        return self._target_column  # プライベート変数から目標列名を返す

    @property
    def __name__(self) -> str:
        # クラス名を返すプロパティ
        return self.__class__.__name__  # 現在のクラスの名前を返す

    @property
    def name(self):
        # クラス名を取得するためのプロパティ（__name__を参照）
        return self.__name__  # クラス名を返す

    @property
    def names(self):
        # 名前のリストを取得するプロパティ
        return self._names  # プライベート変数から名前のリストを返す

    def compute(self):
        # インジケーターの計算を行うメソッド
        raise NotImplementedError  # サブクラスで実装が必要であることを示すエラーを発生させる

    def default_render_options(self):
        # デフォルトのレンダリングオプションを返すメソッド
        return {}  # 空の辞書を返す（オプションが無い場合）

    def render_options(self):
        # レンダリングオプションのコピーを返すメソッド
        return {name: option.copy() for name, option in self._render_options.items()}  # 各オプションのコピーを辞書形式で返す

    def __getitem__(self, index: int):
        # インデックスを指定してデータを取得するメソッド
        row = self._data.iloc[index]  # 指定されたインデックスの行を取得
        for name in self.names:  # 各インジケーター名に対してループ
            if pd.isna(row[name]):  # 行の値がNaN（欠損値）であれば
                return None  # Noneを返す（値がないため）

            self.values[name] = row[name]  # 現在のインジケーター名に対応する値を保存
            if self._render_options.get(name):  # レンダリングオプションが存在する場合
                self._render_options[name].value = row[name]  # レンダリングオプションの値を更新

        return self.serialise()  # データをシリアライズして返す

    def __call__(self, index: int):
        # インデックスを指定して呼び出すとデータを取得するメソッド
        return self[index]  # __getitem__メソッドを呼び出す

    def serialise(self):
        # 現在のインジケーターの状態をシリアライズして辞書形式で返すメソッド
        return {
            'name': self.name,  # インジケーターの名前
            'names': self.names,  # インジケーター名のリスト
            'values': self.values.copy(),  # 現在の値のコピー
            'target_column': self.target_column,  # 目標列名
            'render_options': self.render_options(),  # レンダリングオプションの取得
            'min': self.min,  # 最小値
            'max': self.max  # 最大値
        }

    def config(self):
        # インジケーターの設定を辞書形式で返すメソッド
        return {
            'name': self.name,  # インジケーターの名前
            'names': self.names,  # インジケーター名のリスト
            'target_column': self.target_column,  # 目標列名
            'min': self.min,  # 最小値
            'max': self.max  # 最大値
        }

class SMA(Indicator):
    """ Trend indicator
    トレンドインジケーター

    シンプル移動平均（SMA）は、選択した価格範囲（通常は終値）の平均を、その範囲内の期間数で計算します。

    SMAは、資産価格が強気トレンドまたは弱気トレンドを継続するか逆転するかを判定するためのテクニカルインジケーターです。
    SMAは、株の終値を一定期間合計し、それを観察している期間の数で割ることによって計算されます。
    短期平均は基となる価格の変化に素早く反応し、長期平均は反応が遅くなります。

    https://www.investopedia.com/terms/s/sma.asp
    """
    
    def __init__(
            self, 
            data: pd.DataFrame,  # データのDataFrame
            period: int=20,      # 移動平均期間（デフォルトは20）
            target_column: str='close',  # 関連する価格情報（デフォルトは終値）
            render_options: dict={},      # 表示オプション
            **kwargs
        ):
        self._period = period  # 移動平均期間を初期化
        self._names = [f'SMA{period}']  # SMAの名前設定（例：SMA20）
        
        super().__init__(data, target_column, render_options, **kwargs)  # 基底クラスの初期化
        
        self.min = self._data[self._names[0]].min()  # 最小値を計算
        self.max = self._data[self._names[0]].max()  # 最大値を計算
    
    def default_render_options(self):
        # デフォルトの描画オプションを設定
        return {name: RenderOptions(
            name=name,  # インジケーターの名前
            color=(100, 100, 255),  # RGBカラー
            window_type=WindowType.MAIN,  # 表示ウィンドウタイプ
            render_type=RenderType.LINE,  # 表示タイプ（ライン）
            min=self.min,  # 最小値
            max=self.max   # 最大値
        ) for name in self._names}

    def compute(self):
        # SMAを計算してデータに追加
        self._data[self._names[0]] = self._data[self.target_column].rolling(self._period).mean()

    def config(self):
        # 設定を返す
        config = super().config()  # 基底クラスの設定を取得
        config['period'] = self._period  # 期間を設定
        return config


class BolingerBands(Indicator):
    """ Volatility indicator
    ボリンジャーバンド（Bollinger Bands）は、ボラティリティ指標です。

    ボリンジャーバンドは、ジョン・ボリンジャーによって開発された価格エンベロープの一種です。（価格エンベロープは、上限と下限の価格範囲を定義します。）
    
    ボリンジャーバンドは、価格の単純移動平均の上と下に標準偏差のレベルでプロットされたエンベロープです。バンドの距離は標準偏差に基づいているため、基となる価格のボラティリティの変動に適応します。

    ボリンジャーバンドは、期間と標準偏差（StdDev）の2つのパラメータを使用します。デフォルト値は、期間が20、標準偏差が2ですが、組み合わせはカスタマイズ可能です。

    ボリンジャーバンドは、価格が相対的に高いか低いかを判断するのに役立ちます。上部バンドと下部バンドのペアで使用され、移動平均と一緒に使用されます。さらに、このペアは単独で使用することを意図していません。他のインジケーターからの信号を確認するために使用してください。
    """
    
    def __init__(
            self, 
            data: pd.DataFrame,  # データのDataFrame
            period: int=20,      # ボリンジャーバンドの期間（デフォルトは20）
            std: int=2,          # 標準偏差の倍数（デフォルトは2）
            target_column: str='close',  # 関連する価格情報（デフォルトは終値）
            render_options: dict={},      # 表示オプション
            **kwargs
        ):
        self._period = period  # ボリンジャーバンドの期間を初期化
        self._std = std        # 標準偏差の倍数を初期化
        self._names = ['SMA', 'BB_up', 'BB_dn']  # インジケーターの名前を設定
        super().__init__(data, target_column, render_options, **kwargs)  # 基底クラスの初期化
        
        self.min = self._data['BB_dn'].min()  # 下部バンドの最小値を計算
        self.max = self._data['BB_up'].max()  # 上部バンドの最大値を計算

    def compute(self):
        # SMA、上部バンド、下部バンドを計算
        self._data['SMA'] = self._data[self.target_column].rolling(self._period).mean()  # SMAを計算
        self._data['BB_up'] = self._data['SMA'] + self._data[self.target_column].rolling(self._period).std() * self._std  # 上部バンドを計算
        self._data['BB_dn'] = self._data['SMA'] - self._data[self.target_column].rolling(self._period).std() * self._std  # 下部バンドを計算

    def default_render_options(self):
        # デフォルトの描画オプションを設定
        return {name: RenderOptions(
            name=name,  # インジケーターの名前
            color=(100, 100, 255),  # RGBカラー
            window_type=WindowType.MAIN,  # 表示ウィンドウタイプ
            render_type=RenderType.LINE,  # 表示タイプ（ライン）
            min=self.min,  # 最小値
            max=self.max   # 最大値
        ) for name in self._names}

    def config(self):
        # 設定を返す
        config = super().config()  # 基底クラスの設定を取得
        config['period'] = self._period  # 期間を設定
        config['std'] = self._std  # 標準偏差を設定
        return config

class RSI(Indicator):
    """ Momentum indicator
    モメンタム指標

    相対力指数（RSI）は、J. ウェルズ・ワイルダーによって開発されたモメンタムオシレーターで、価格変動の速度と変化を測定します。
    RSIは0から100の間で変動し、従来は70以上で買われすぎ、30以下で売られすぎと見なされます。
    シグナルは、ダイバージェンスや失敗のスイングを探ることによって生成できます。
    
    RSIは、一般的なトレンドを特定するためにも使用できます。
    """
    
    def __init__(
            self, 
            data: pd.DataFrame,  # データのDataFrame
            period: int=14,      # RSIの計算期間（デフォルトは14）
            target_column: str='close',  # 関連する価格情報（デフォルトは終値）
            render_options: dict={},      # 表示オプション
            min: float=0.0,      # 最小値（デフォルトは0.0）
            max: float=100.0,    # 最大値（デフォルトは100.0）
            **kwargs
        ):
        self._period = period  # RSIの計算期間を初期化
        self._names = ['RSI']  # インジケーターの名前を設定
        super().__init__(data, target_column, render_options, min=min, max=max, **kwargs)  # 基底クラスの初期化

    def compute(self):
        # RSIを計算するメソッド
        delta = self._data[self.target_column].diff()  # 価格の変化を計算

        up = delta.clip(lower=0)  # 上昇幅を取得（負の値は0にクリップ）
        down = -1 * delta.clip(upper=0)  # 降下幅を取得（正の値は0にクリップ）

        # 指数移動平均を計算
        ema_up = up.ewm(com=self._period-1, adjust=True, min_periods=self._period).mean()  # 上昇幅のEMA
        ema_down = down.ewm(com=self._period-1, adjust=True, min_periods=self._period).mean()  # 降下幅のEMA

        rs = ema_up / ema_down  # RS（相対強度）を計算
        self._data['RSI'] = 100 - (100 / (1 + rs))  # RSIを計算してデータに追加

    def default_render_options(self):
        # デフォルトの描画オプションを設定
        custom_options = {
            "RSI0": 0,   # RSIの0ライン
            "RSI30": 30, # RSIの30ライン
            "RSI70": 70, # RSIの70ライン
            "RSI100": 100 # RSIの100ライン
        }
        
        # デフォルトのオプションを生成
        options = {name: RenderOptions(
            name=name,  # インジケーターの名前
            color=(100, 100, 255),  # RGBカラー
            window_type=WindowType.SEPERATE,  # 表示ウィンドウタイプ
            render_type=RenderType.LINE,  # 表示タイプ（ライン）
            min=self.min,  # 最小値
            max=self.max   # 最大値
        ) for name in self._names}

        # カスタムオプションを追加
        for name, value in custom_options.items():
            options[name] = RenderOptions(
                name=name,  # カスタムラインの名前
                color=(192, 192, 192),  # RGBカラー（グレー）
                window_type=WindowType.SEPERATE,  # 表示ウィンドウタイプ
                render_type=RenderType.LINE,  # 表示タイプ（ライン）
                min=self.min,  # 最小値
                max=self.max,   # 最大値
                value=value    # ラインの値
            )
        return options  # 設定したオプションを返す

    def config(self):
        # 設定を返すメソッド
        config = super().config()  # 基底クラスの設定を取得
        config['period'] = self._period  # 期間を設定
        return config  # 設定を返す

class PSAR(Indicator):
    """ Parabolic Stop and Reverse (Parabolic SAR)
    パラボリック・ストップ・アンド・リバース（パラボリックSAR）

    パラボリック・ストップ・アンド・リバース（通称パラボリックSAR）は、J. Welles Wilderが開発したトレンドフォローのインジケーターです。

    パラボリックSARは、上昇トレンドでは価格バーの下に、下降トレンドでは価格バーの上に表示される点または曲線です。

    https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar
    """
    
    def __init__(
            self, 
            data: pd.DataFrame,  # データを含むPandas DataFrame
            step: float=0.02,    # 加速因子（デフォルトは0.02）
            max_step: float=0.2, # 最大加速因子（デフォルトは0.2）
            target_column: str='close',  # 関連する価格情報（デフォルトは終値）
            render_options: dict={},  # 描画オプション
            **kwargs
        ):
        self._names = ['PSAR']  # PSARインジケーターの名称
        self._step = step  # 加速因子を設定（トレンド強度に応じてPSARの移動速度を調整）
        self._max_step = max_step  # 最大加速因子を設定（過度な変化を防ぐための上限）
        
        # 基底クラスの初期化
        super().__init__(data, target_column, render_options, **kwargs)
        
        # 最小および最大値を計算(PSARの描画範囲を決定)
        self.min = self._data['PSAR'].min()  
        self.max = self._data['PSAR'].max()  

    def default_render_options(self):
        # デフォルトの描画オプションを設定
        return {name: RenderOptions(
            name=name,  # インジケーターの名前
            color=(100, 100, 255),  # 点の色（RGB形式）
            window_type=WindowType.MAIN,  # 表示ウィンドウのタイプ
            render_type=RenderType.DOT,  # 描画タイプ（ドット）
            min=self.min,  # 最小値
            max=self.max   # 最大値
        ) for name in self._names}

    def compute(self):
        # PSARを計算するメソッド
        high = self._data['high']  # 高値
        low = self._data['low']    # 安値
        close = self._data[self.target_column]  # 終値

        up_trend = True  # 現在のトレンドが上昇かどうかを示すフラグ
        acceleration_factor = self._step  # 加速因子を初期化(トレンドが強まるにつれて増加)
        up_trend_high = high.iloc[0]  # 上昇トレンドの高値の初期値
        down_trend_low = low.iloc[0]   # 下降トレンドの安値の初期値

        self._psar = close.copy()  # PSARを計算するためのT列をコピー
        self._psar_up = pd.Series(index=self._psar.index, dtype="float64")  # 上昇トレンドのPSARを格納するSeries
        self._psar_down = pd.Series(index=self._psar.index, dtype="float64")  # 下降トレンドのPSARを格納するSeries

        # 価格データのループ処理
        for i in range(2, len(close)):
            reversal = False  # 反転のフラグ

            max_high = high.iloc[i]  # 現在の高値を取得
            min_low = low.iloc[i]     # 現在の安値を取得

            if up_trend:  # 上昇トレンドの場合
                self._psar.iloc[i] = self._psar.iloc[i - 1] + (
                    acceleration_factor * (up_trend_high - self._psar.iloc[i - 1])
                )  # PSARの値を加速因子を使って更新

                # 反転条件
                if min_low < self._psar.iloc[i]:  # 現在の安値がPSARを下回った場合
                    reversal = True  # 反転フラグを立てる
                    self._psar.iloc[i] = up_trend_high  # PSARを上昇トレンドの高値に設定
                    down_trend_low = min_low  # 下降トレンドの安値を更新
                    acceleration_factor = self._step  # 加速因子をリセット
                else:
                    if max_high > up_trend_high:  # 高値を更新する場合
                        up_trend_high = max_high   # 上昇トレンドの高値を更新
                        acceleration_factor = min(
                            acceleration_factor + self._step, self._max_step
                        )  # 加速因子を更新（最大に制限）

                    # PSARの下限を設定
                    low1 = low.iloc[i - 1]
                    low2 = low.iloc[i - 2]
                    # PSARの下限を設定
                    if low2 < self._psar.iloc[i]:  # 前の安値がPSARより低ければ設定
                        self._psar.iloc[i] = low2
                    elif low1 < self._psar.iloc[i]:  # 1つ前の安値がPSARより低ければ設定
                        self._psar.iloc[i] = low1
            else:  # 下降トレンドの場合
                self._psar.iloc[i] = self._psar.iloc[i - 1] - (
                    acceleration_factor * (self._psar.iloc[i - 1] - down_trend_low)
                )  # PSARの値を加速因子を使って更新

                # 反転条件
                if max_high > self._psar.iloc[i]:  # 現在の高値がPSARを上回った場合
                    reversal = True  # 反転フラグを立てる
                    self._psar.iloc[i] = down_trend_low  # PSARを下降トレンドの安値に設定
                    up_trend_high = max_high  # 上昇トレンドの高値を更新
                    acceleration_factor = self._step  # 加速因子をリセット
                else:
                    if min_low < down_trend_low:  # 安値を更新する場合
                        down_trend_low = min_low   # 下降トレンドの安値を更新
                        acceleration_factor = min(
                            acceleration_factor + self._step, self._max_step
                        )  # 加速因子を更新（最大に制限）
                    
                    # PSARの上限を設定
                    high1 = high.iloc[i - 1]
                    high2 = high.iloc[i - 2]
                    # PSARの上限を設定
                    if high2 > self._psar.iloc[i]:  # 前の高値がPSARより高ければ設定
                        self._psar[i] = high2
                    elif high1 > self._psar.iloc[i]:  # １つ前の高値がPSARより高ければ設定
                        self._psar.iloc[i] = high1

            # トレンドの反転を判定（XOR）
            up_trend = up_trend != reversal   # 反転があればトレンドを更新  

            # トレンドに基づいてPSARを保存
            if up_trend:
                self._psar_up.iloc[i] = self._psar.iloc[i]  # 上昇トレンドPSARを保存
            else:
                self._psar_down.iloc[i] = self._psar.iloc[i]  # 下降トレンドPSARを保存

        # PSARインジケーターをデータに計算
        self._data['PSAR'] = self._psar   # 計算したPSARをデータフレームに追加

    def config(self):
        config = super().config()  # 基底クラスの設定を取得
        config['step'] = self._step  # 加速因子を設定
        config['max_step'] = self._max_step  # 最大加速因子を設定
        return config

class MACD(Indicator):
    """ Moving Average Convergence Divergence (MACD)
    移動平均収束発散（MACD）

    MACDは、トレンドの強さと方向を示すテクニカル指標であり、短期と長期の指数移動平均（EMA）の差を利用して計算されます。
    """

    def __init__(
            self, 
            data: pd.DataFrame,  # データを含むPandas DataFrame
            fast_ma: int = 12,   # 短期移動平均の期間（デフォルトは12）
            slow_ma: int = 26,   # 長期移動平均の期間（デフォルトは26）
            histogram: int = 9,   # ヒストグラムの期間（デフォルトは9）
            target_column: str='close',  # 関連する価格情報（デフォルトは終値）
            render_options: dict={},  # 描画オプション
            **kwargs
        ):
        self._fast_ma = fast_ma  # 短期移動平均の期間を初期化
        self._slow_ma = slow_ma  # 長期移動平均の期間を初期化
        self._histogram = histogram  # ヒストグラムの期間を初期化
        self._names = ['MACD', 'MACD_signal']  # MACDとシグナルラインの名前を設定
        
        # 基底クラスの初期化
        super().__init__(data, target_column, render_options, **kwargs)
        
        # シグナルラインの最小値と最大値を計算
        self.min = self._data['MACD_signal'].min()  
        self.max = self._data['MACD_signal'].max()  

    def compute(self):
        # 短期の指数移動平均（EMA）を計算
        short_ema = self._data[self.target_column].ewm(span=self._fast_ma, adjust=False).mean()
        
        # 長期の指数移動平均（EMA）を計算
        long_ema = self._data[self.target_column].ewm(span=self._slow_ma, adjust=False).mean()

        # 移動平均収束発散（MACD）を計算
        self._data["MACD"] = short_ema - long_ema  # 短期EMAから長期EMAを引く

        # シグナルラインを計算
        self._data["MACD_signal"] = self._data["MACD"].ewm(span=self._histogram, adjust=False).mean()  # MACDのEMAを計算

    def default_render_options(self):
        # デフォルトの描画オプションを設定
        return {name: RenderOptions(
            name=name,  # インジケーターの名前
            color=(100, 100, 255),  # 描画色（RGB形式）
            window_type=WindowType.SEPERATE,  # 表示ウィンドウのタイプ
            render_type=RenderType.LINE,  # 描画タイプ（ライン）
            min=self.min,  # 最小値
            max=self.max   # 最大値
        ) for name in self._names}
    
    def config(self):
        # 設定を返すメソッド
        config = super().config()  # 基底クラスの設定を取得
        config['fast_ma'] = self._fast_ma  # 短期移動平均の期間を設定
        config['slow_ma'] = self._slow_ma  # 長期移動平均の期間を設定
        config['histogram'] = self._histogram  # ヒストグラムの期間を設定
        return config  # 設定を返す
   