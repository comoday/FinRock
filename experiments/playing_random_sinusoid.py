import numpy as np
import pandas as pd
# finrockパッケージから必要なクラスをインポート
from finrock.data_feeder import PdDataFeeder
from finrock.trading_env import TradingEnv
from finrock.render import PygameRender
from finrock.scalers import ZScoreScaler  # Zスコアによるスケーリングクラス
from finrock.reward import AccountValueChangeReward
from finrock.indicators import BolingerBands, SMA, RSI, PSAR, MACD
from finrock.metrics import DifferentActions, AccountValue, MaxDrawdown, SharpeRatio

# CSVファイルからデータを読み込む
df = pd.read_csv('Datasets/random_sinusoid.csv')

# PdDataFeederインスタンスを作成
pd_data_feeder = PdDataFeeder(
    df = df,
    # インジケータの追加
    indicators = [
        BolingerBands(data=df, period=20, std=2),
        RSI(data=df, period=14),
        PSAR(data=df),
        MACD(data=df),
        SMA(data=df, period=7),
    ]
)

# TradingEnvインスタンスを作成(エージェントが取引を行うシミュレーション)
env = TradingEnv(
    data_feeder = pd_data_feeder,  # データ供給者としてPdDataFeederを指定
    output_transformer = ZScoreScaler(),  # Zスコアスケーラーを使用(標準正規分布に変換)
    initial_balance = 1000.0,   # 初期バランスを1000.0に設定
    max_episode_steps = 1000,   # エピソード最大ステップ数を1000に設定
    window_size = 50,   # 環境のウィンドウサイズを50に設定
    reward_function = AccountValueChangeReward(),  # 報酬関数としてアカウント価値変化報酬を使用
    metrics = [
        # パフォーマンス指標のリスト
        DifferentActions(),  # 異なるアクションの数を測定
        AccountValue(),      # アカウントの価値を測定
        MaxDrawdown(),       # 最大ドローダウンを測定
        SharpeRatio(),       # シャープレシオを測定
    ]
)

# 環境のアクション空間を取得(可能なアクションの数)
action_space = env.action_space
# 環境の観察空間の形状を取得(次元の数)
input_shape = env.observation_space.shape

# 環境設定をJSONファイルとして保存
env.save_config()

# Pygameを使用して描画するためのレンダラーを作成
pygameRender = PygameRender(frame_rate=60)  # フレームレートを60に設定

# 環境をリセットして初期状態と情報を取得
state, info = env.reset()
pygameRender.render(info)  # 初期情報を描画

# 報酬の累積値を初期化
rewards = 0.0
# 無限ループでエピソードを実行
while True:
    # モデルの予測をシミュレートし、ランダムなアクションを選択
    action = np.random.randint(0, action_space)  # 0 からアクション空間のサイズまでのランダムな整数を生成

    # 環境にアクションを適用し、次の状態、報酬、終了フラグ、トランケートフラグ、情報を取得
    state, reward, terminated, truncated, info = env.step(action)
    rewards += reward  # 報酬を累積(累積報酬を更新)

    pygameRender.render(info)  # 情報を描画

    # 環境が終了またはトランケート(打ち切り)された場合
    if terminated or truncated:
        # 最終アカウント価値と累積報酬を出力
        print(info['states'][-1].account_value, rewards)
        rewards = 0.0  # 報酬をリセット
        state, info = env.reset()  # 環境をリセット
        pygameRender.reset()   # レンダラー(描画)をリセット