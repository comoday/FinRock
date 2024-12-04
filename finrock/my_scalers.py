import numpy as np
np.seterr(all="ignore")   # 全ての警告を無視する設定
import warnings
from my_state import Observations


class Scaler:
    def __init__(self):
        # Scalerクラスの初期化メソッド（特に何も初期化しない）
        pass
    
    def transform(self, observations: Observations) -> np.ndarray:
        # 観測データを変換するメソッド（サブクラスで実装が必要）
        raise NotImplementedError
    
    def __call__(self, observations) -> np.ndarray:
        # インスタンスを関数のように呼び出すメソッド
        assert isinstance(observations, Observations) == True, "observations must be an instance of Observations"
        # 引数がObservationsのインスタンスであることを確認
        return self.transform(observations)  # 観測データを変換するメソッドを呼び出す
    
    @property
    def __name__(self) -> str:
        # クラスの名前を返すプロパティ
        return self.__class__.__name__

    @property
    def name(self) -> str:
        # クラスの名前を返すためのプロパティ(__name__を呼び出す)
        return self.__name__


class MinMaxScaler(Scaler):
    def __init__(self, min: float, max: float):
        # MinMaxScalerクラスの初期化メソッド
        super().__init__()  # 親クラスScalerの初期化を呼び出す
        self._min = min  # スケーリングの最小値を設定
        self._max = max  # スケーリングの最大値を設定
    
    def transform(self, observations: Observations) -> np.ndarray:
        # 観測データをMin-Maxスケーリングで変換するメソッド
        transformed_data = []  # 変換後のデータを格納するリストを初期化
        for state in observations:  # 各状態に対してループ
            data = []  # 各状態の変換結果を格納するリストを初期化
            for name in ['open', 'high', 'low', 'close']:  # 必要なフィールドに対してループ
                value = getattr(state, name)  # 状態から対応する値を取得
                # Min-Maxスケーリングを適用
                transformed_value = (value - self._min) / (self._max - self._min)
                data.append(transformed_value)  # 変換された値をリストに追加
            
            data.append(state.allocation_percentage)  # 割り当て割合を追加

            # スケーリングされたインジケーターを追加
            for indicator in state.indicators:  # 各インジケーターに対してループ
                for value in indicator["values"].values():  # インジケーター内の値に対してループ
                    # Min-Maxスケーリングを適用
                    transformed_value = (value - indicator["min"]) / (indicator["max"] - indicator["min"])
                    data.append(transformed_value)  # 変換されたインジケーターの値をリストに追加

            transformed_data.append(data)  # 各状態の変換結果を全体のリストに追加

        results = np.array(transformed_data)  # リストをNumPy配列に変換

        return results  # 変換されたデータを返す
   
class ZScoreScaler(Scaler):
    def __init__(self):
        # ZScoreScalerクラスの初期化メソッド
        super().__init__()  # 親クラスScalerの初期化を呼び出す
        # RuntimeWarningの警告を無視する設定（特にオーバーフローに関するもの）
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in reduce")
    
    def transform(self, observations: Observations) -> np.ndarray:
        # 観測データをZスコアスケーリングで変換するメソッド
        full_data = []  # すべてのデータを格納するリストを初期化

        print("Observations:", observations)   # 内容確認 
        
        for state in observations:  # 各状態に対してループ
            # 必要なフィールド（始値、高値、安値、終値、割り当て割合）を取得
            data = [getattr(state, name) for name in ['open', 'high', 'low', 'close', 'allocation_percentage']]

            print("State:", state)   # 内容確認

            for indicator in state.indicators:
                print("Indicator:", indicator)   # 内容確認
                # indicator["values"]が辞書であることを確認
                if isinstance(indicator["values"], dict):
                    data += [value for value in indicator["values"].values()]
                else:
                    print("Error: indicator['values] is not a dictionary:", indicator["values"])

            full_data.append(data)  # 各状態のデータを全体のリストに追加

        results = np.array(full_data)  # リストをNumPy配列に変換

        # 収益率を計算（前の行との差分を取る）し、ゼロで割った場合にNaNをゼロに変換
        returns = np.nan_to_num(np.diff(results, axis=0) / results[:-1])

        # Zスコアを計算（収益率から平均を引き、標準偏差で割る）
        z_scores = np.nan_to_num((returns - np.mean(returns, axis=0)) / np.std(returns, axis=0))

        return z_scores  # Zスコアを返す
