import numpy as np
from my_state import Observations

class Reward:
    def __init__(self) -> None:  # Rewardクラスの初期化メソッド
        pass

    @property
    def __name__(self) -> str:    # クラス名を返すプロパティ
        return self.__class__.__name__
    
    def __call__(self, observations: Observations) -> float:
        # 観測データを受け取って報酬を計算するメソッド（サブクラスで実装が必要）
        raise NotImplementedError
    
    def reset(self, observations: Observations):
        # リセットメソッド（特に何もしない）
        pass
    

class SimpleReward(Reward):
    def __init__(self) -> None:    # SimpleRewardクラスの初期化メソッド
        super().__init__()   # 親クラスRewardの初期化を呼び出す

    def __call__(self, observations: Observations) -> float:
        # 観測データを受け取り、報酬を計算するメソッド
        if not isinstance(observations, Observations):
            raise ValueError("observations must be an instance of Observations")
        
        # 観測から最後の2つの状態を取得
        if len(observations.observations) < 2:
            return 0.0  # 観測が不十分な場合は報酬を0にする
        
        last_state, next_state = observations[-2:]

        # 購入アクションの場合
        if next_state.allocation_percentage > last_state.allocation_percentage:
            # 購入が良かったか悪かったかを判断
            order_size = next_state.allocation_percentage - last_state.allocation_percentage   # 購入サイズを計算
            if last_state.close == 0:  # ゼロ除算を防ぐ
                return 0.0
            # 購入による報酬を計算
            reward = (next_state.close - last_state.close) / last_state.close * order_size
            # 保持による報酬を計算
            hold_reward = (next_state.close - last_state.close) / last_state.close * last_state.allocation_percentage
            reward += hold_reward  # 購入報酬と保持報酬を合算

        # 売却アクションの場合
        elif next_state.allocation_percentage < last_state.allocation_percentage:
            # 売却が良かったか悪かったかを判断
            order_size = last_state.allocation_percentage - next_state.allocation_percentage   # 売却サイズを計算
            if last_state.close == 0:  # ゼロ除算を防ぐ
                return 0.0
            # 売却による報酬を計算
            reward = -1 * (next_state.close - last_state.close) / last_state.close * order_size
            # 保持による報酬を計算
            hold_reward = (next_state.close - last_state.close) / last_state.close * next_state.allocation_percentage
            reward += hold_reward   # 売却報酬と保持報酬を合算

        # 保持アクションの場合
        else:
            # 保持かよかったか悪かったかを判断
            ratio = -1 if not last_state.allocation_percentage else last_state.allocation_percentage  # 割り当て割合を取得
            if last_state.close == 0:
                return 0.0
            reward = (next_state.close - last_state.close) / last_state.close * ratio  # 保持による報酬を計算
            
        return reward   # 計算された報酬を返す

class AccountValueChangeReward(Reward):
    def __init__(self) -> None:  # AccountValueChangeRewardクラスの初期化メソッド
        super().__init__()       # 親クラスRewardの初期化を呼び出す
        self.ratio_days=365.25   # 日数の比率を設定（通常の年数、うるう年を考慮）


    def reset(self, observations: Observations):   # リセットメソッド（状態をリセット）
        super().reset(observations)                # 親クラスのリセットメソッドを呼び出す
        self.returns = []                          # 収益率のリストを初期化
    
    def __call__(self, observations: Observations) -> float:
        # 観測データを受け取り、報酬を計算するメソッド
        assert isinstance(observations, Observations) == True, "observations must be an instance of Observations"

        # 観測から最後の2つの状態を取得
        last_state, next_state = observations[-2:]
        # アカウントの価値の変化に基づく報酬を計算
        reward = (next_state.account_value - last_state.account_value) / last_state.account_value

        return reward    # 計算された報酬を返す