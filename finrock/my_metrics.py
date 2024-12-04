from my_state import State  # Stateクラスを同じパッケージからインポート
import numpy as np  # NumPyをインポート。数値計算や配列操作に使用。

""" 
Metricsは環境に関する情報を追跡し、記録するために使用されます。
可能なメトリクスのリスト:
+ DifferentActions,  # 異なるアクションの数を追跡
+ AccountValue,      # アカウントの価値を追跡
+ MaxDrawdown,      # 最大ドローダウンを計算
+ SharpeRatio,      # シャープレシオを計算
- AverageProfit,    # 平均利益を計算
- AverageLoss,      # 平均損失を計算
- AverageTrade,     # 平均取引額を計算
- WinRate,          # 勝率を計算
- LossRate,         # 損失率を計算
- AverageWin,       # 平均勝ち額を計算
- AverageLoss,      # 平均負け額を計算
- AverageWinLossRatio,  # 勝ち/負け比率を計算
- AverageTradeDuration,  # 平均取引期間を計算
- AverageTradeReturn,     # 平均取引リターンを計算
"""

class Metric:
    def __init__(self, name: str="metric") -> None:
        # Metricクラスの初期化メソッド
        self.name = name  # メトリクスの名前を設定（デフォルトは"metric"）
        self.reset()  # メトリクスを初期状態にリセットするメソッドを呼び出す

    @property
    def __name__(self) -> str:
        # クラス名を返すプロパティ
        return self.__class__.__name__  # クラスの名前を取得して返す

    def update(self, state: State):
        # メトリクスを更新するメソッド
        assert isinstance(state, State), f'state must be State, received: {type(state)}'
        # 引数がStateのインスタンスであることを確認。そうでない場合はエラーメッセージを表示。

        return state  # 引数の状態をそのまま返す（後でメトリクスの計算に使用可能）

    @property
    def result(self):
        # メトリクスの計算結果を返すプロパティ
        raise NotImplementedError  # サブクラスで実装が必要であることを示すエラーを発生させる

    def reset(self, prev_state: State=None):
        # メトリクスの状態をリセットするメソッド
        assert prev_state is None or isinstance(prev_state, State), f'prev_state must be None or State, received: {type(prev_state)}'
        # prev_stateがNoneまたはStateのインスタンスであることを確認。そうでない場合はエラーメッセージを表示。

        return prev_state  # 前の状態を返す（リセット時に必要な場合がある）

# Metricクラスを継承したDifferentActionaクラス(異なる行動のメトリックを追跡)    
class DifferentActions(Metric):
    # コンストラクタでメトリックの名前を設定
    def __init__(self, name: str="different_actions") -> None:
        super().__init__(name=name)   # 親クラスの初期化メソッドを呼び出す

    # 状態を受け取り、異なる行動を更新するメソッド
    def update(self, state: State):
        super().update(state)   # 親クラスのupdateメソッドを呼び出す

        if not self.prev_state:  # prev_stateが未定義の場合
            self.prev_state = state  # 現在の状態をprev_stateとして保存
        else:
            # 現在の状態の割り当て割合が前の状態と異なる場合
            if state.allocation_percentage != self.prev_state.allocation_percentage:
                self.different_actions += 1  # 異なる行動のカウントを増やす

            self.prev_state = state  # 現在の状態をprev_stateとして更新

    @property
    # 異なる行動の数を返すプロパティ
    def result(self):
        return self.different_actions
    
    # メトリックのリセットを行うメソッド
    def reset(self, prev_state: State=None):
        super().reset(prev_state)  # 親クラスのresetメソッドを呼び出す

        self.prev_state = prev_state  # 前の状態を保存
        self.different_actions = 0  # 異なる行動のカウントを0に戻す
        
# Metricクラスを継承したAccountValueクラス(アカウントの総価値を追跡)
class AccountValue(Metric):
    def __init__(self, name: str="account_value") -> None:
        # AccountValueメトリクスの初期化メソッド
        super().__init__(name=name)  # 親クラスMetricの初期化を呼び出す

    def update(self, state: State):
        # メトリクスを更新するメソッド
        super().update(state)  # 親クラスのupdateメソッドを呼び出す

        self.account_value = state.account_value  # 現在のアカウントの価値を取得して保存

    @property
    def result(self):
        # 現在のアカウントの価値を返すプロパティ
        return self.account_value  # 保存されたアカウントの価値を返す
    
    def reset(self, prev_state: State=None):
        # メトリクスの状態をリセットするメソッド
        super().reset(prev_state)  # 親クラスのresetメソッドを呼び出す
        
        # 前の状態があればそのアカウントの価値を設定、なければ0.0に初期化
        self.account_value = prev_state.account_value if prev_state else 0.0


class MaxDrawdown(Metric):
    """ 
    最大ドローダウン（MDD）は、特定の期間内にポートフォリオや投資の価値が
    最大から最小までどれだけ減少したかを測定する指標です。

    最大ドローダウン比率は、最大減少時に失われたピーク値の割合を表します。
    これは、特定の投資やポートフォリオに関連するリスクの指標です。
    投資家やファンドマネージャーは、最大ドローダウンとその比率を使用して
    過去の下振れリスクや潜在的な損失を評価します。
    """
    # コンストラクタでメトリックの名前を設定
    def __init__(self, name: str="max_drawdown") -> None:
        # MaxDrawdownメトリクスの初期化メソッド
        super().__init__(name=name)  # 親クラスMetricの初期化を呼び出す
        self.max_account_value = 0.0  # 最大アカウントの価値を初期化
        self.max_drawdown = 0.0  # 最大ドローダウンを初期化

    # 状態を受け取り、最大ドローダウンを更新するメソッド
    def update(self, state: State):
        # メトリクスを更新するメソッド
        super().update(state)  # 親クラスのupdateメソッドを呼び出す

        # 現在のアカウントの価値とこれまでの最大アカウントの価値を比較し、最大値を更新
        self.max_account_value = max(self.max_account_value, state.account_value)

        # ドローダウンを計算
        drawdown = (state.account_value - self.max_account_value) / self.max_account_value

        # 現在のドローダウンがこれまでの最大ドローダウンよりも大きい場合、更新
        self.max_drawdown = min(self.max_drawdown, drawdown)

    @property
    def result(self):
        # 現在の最大ドローダウンを返すプロパティ
        return self.max_drawdown  # 計算された最大ドローダウンを返す
    
    def reset(self, prev_state: State=None):
        # メトリクスの状態をリセットするメソッド
        super().reset(prev_state)  # 親クラスのresetメソッドを呼び出す

        # 前の状態があればそのアカウントの価値を設定、なければ0.0に初期化
        self.max_account_value = prev_state.account_value if prev_state else 0.0
        self.max_drawdown = 0.0  # 最大ドローダウンをリセット

class SharpeRatio(Metric):
    """ 
    シャープレシオ（Sharpe Ratio）は、投資またはポートフォリオのリスク調整後のパフォーマンスを測定する指標です。
    これは、投資のリターンをリスクに対して評価するのに役立ちます。

    シャープレシオが高いほど、リスク調整後のパフォーマンスが良好であることを示します。
    投資家やポートフォリオマネージャーは、異なる投資やポートフォリオのリスク調整後のリターンを比較するために
    シャープレシオを使用します。これにより、追加のリスクを取ることで得られる追加のリターンが正当化されるかを評価できます。
    """
    # コンストラクタでシャープレシオを計算するためのパラメータを設定
    def __init__(self, ratio_days=365.25, name: str='sharpe_ratio'):
        # SharpeRatioメトリクスの初期化メソッド
        self.ratio_days = ratio_days  # 年間の日数を設定（デフォルトは365.25日）
        super().__init__(name=name)  # 親クラスMetricの初期化を呼び出す

    # 状態を受け取り、シャープレシオを更新するメソッド
    def update(self, state: State):
        # メトリクスを更新するメソッド
        super().update(state)  # 親クラスのupdateメソッドを呼び出す

        # 現在の状態の日付と前の状態の日付の差を計算
        time_difference_days = (state.date - self.prev_state.date).days
        
        # 1日以上の差があれば日次リターンを計算
        if time_difference_days >= 1:
            # 日次リターンを計算しリストに追加
            self.daily_returns.append((state.account_value - self.prev_state.account_value) / self.prev_state.account_value)
            self.prev_state = state  # 現在の状態を前の状態として更新
        
    @property
    def result(self):
        # メトリクスの計算結果を返すプロパティ
        if len(self.daily_returns) == 0:
            return 0.0  # 日次リターンが空の場合は0.0を返す

        mean = np.mean(self.daily_returns)  # 日次リターンの平均を計算
        std = np.std(self.daily_returns)  # 日次リターンの標準偏差を計算
        
        if std == 0:
            return 0.0  # 標準偏差が0の場合は0.0を返す（変動がないため）

        # シャープレシオを計算
        sharpe_ratio = mean / std * np.sqrt(self.ratio_days)
        
        return sharpe_ratio  # 計算されたシャープレシオを返す
    
    def reset(self, prev_state: State=None):
        # メトリクスの状態をリセットするメソッド
        super().reset(prev_state)  # 親クラスのresetメソッドを呼び出す
        self.prev_state = prev_state  # 前の状態を設定
        self.daily_returns = []  # 日次リターンのリストを初期化
