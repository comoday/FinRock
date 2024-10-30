import numpy as np
import pandas as pd
import tensorflow as tf
# TensorFlowのログレベルをERRORに設定（警告や情報メッセージを表示しない）
tf.get_logger().setLevel('ERROR')
# GPUデバイスのメモリ成長を有効にする設定
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from finrock.data_feeder import PdDataFeeder
from finrock.trading_env import TradingEnv
from finrock.render import PygameRender

# CSVファイルからデータを読み込む
df = pd.read_csv('Datasets/random_sinusoid.csv')
# 最後の1000行を取り出してデータフレームを作成
df = df[-1000:]

# モデルのパスを指定
model_path = "runs/1704746665"

# PdDataFeederの設定をモデルパスから読み込む
pd_data_feeder = PdDataFeeder.load_config(df, model_path)
# TradingEnvの設定をPdDataFeederから読み込む
env = TradingEnv.load_config(pd_data_feeder, model_path)

# 環境からアクション空間と観察空間の形状を取得
action_space = env.action_space    # 環境のアクション空間を取得
input_shape = env.observation_space.shape    # 環境の観察空間の形状を取得

# Pygameを使用して描画するためのレンダラーを作成
pygameRender = PygameRender(frame_rate=120)  # フレームレートを120に設定

# 学習済みモデル（アクターモデル）を読み込む
agent = tf.keras.models.load_model(f'{model_path}/ppo_sinusoid_actor.h5')

# 環境をリセットして初期状態と情報を取得
state, info = env.reset()
# 初期状態の情報を描画
pygameRender.render(info)
rewards = 0.0   # 報酬を初期化

# 無限ループでエピソードを実行
while True:
     # モデルによるアクションの予測
    prob = agent.predict(np.expand_dims(state, axis=0), verbose=False)[0]   # 状態を次元拡張してモデルに入力
    
    """ 予測された確率から最も高いアクションを選択 """
    action = np.argmax(prob)    # 確率が最も高いアクションを選択

     # 環境にアクションを適用し、次の状態、報酬、終了フラグ、トランケートフラグ、情報を取得
    state, reward, terminated, truncated, info = env.step(action)
    # 報酬を累積
    rewards += reward
    # 情報を描画
    pygameRender.render(info)

    # 環境が終了またはトランケート（打ち切り）された場合
    if terminated or truncated:
        print(rewards)   # 累積報酬を出力
        # メトリックの情報を出力
        for metric, value in info['metrics'].items():
            print(metric, value)  # 各メトリックとその値を出力

        # 環境をリセットして新しい状態と情報を取得
        state, info = env.reset()
        rewards = 0.0    # 報酬をリセット
        pygameRender.reset()   # レンダラーをリセット
        pygameRender.render(info)   # 初期状態の情報を描画