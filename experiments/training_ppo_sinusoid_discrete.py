import os
import numpy as np
import pandas as pd
import tensorflow as tf
# TensorFlowのログレベルをERRORに設定(警告や情報メッセージを表示しない)
tf.get_logger().setLevel('ERROR')
# GPUデバイスが存在する場合、メモリの成長を有効にする
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from keras import layers, models

from finrock.data_feeder import PdDataFeeder
from finrock.trading_env import TradingEnv
from finrock.scalers import MinMaxScaler, ZScoreScaler
from finrock.reward import SimpleReward, AccountValueChangeReward
from finrock.metrics import DifferentActions, AccountValue, MaxDrawdown, SharpeRatio
from finrock.indicators import BolingerBands, RSI, PSAR, SMA, MACD

from rockrl.utils.misc import MeanAverage
from rockrl.utils.memory import MemoryManager
from rockrl.tensorflow import PPOAgent
from rockrl.utils.vectorizedEnv import VectorizedEnv

if __name__ == '__main__':
    # CSVファイルからデータを読み込む
    df = pd.read_csv('Datasets/random_sinusoid.csv')
    df = df[:-1000]  # テスト用にラスト1000行を取り除く

    # PdDataFeederインスタンスを作成
    pd_data_feeder = PdDataFeeder(
        df,  # 読み込んだDFを渡す
        # 使用するインジケータのリスト
        indicators = [
            BolingerBands(data=df, period=20, std=2),
            RSI(data=df, period=14),
            PSAR(data=df),
            MACD(data=df),
            SMA(data=df, period=7),
        ]
    )
    # 環境の数を設定
    num_envs = 10
    # ベクトル化された環境のインスタンスを作成
    env = VectorizedEnv(
        env_object = TradingEnv,  # 使用する環境のクラス
        num_envs = num_envs,      # 環境の数を指定
        data_feeder = pd_data_feeder,  # データ供給者としてPdDataFeederを指定
        output_transformer = ZScoreScaler(),  # Zスコアスケーラを使用
        initial_balance = 1000.0,  # 初期バランスを1000.0に設定
        max_episode_steps = 1000,  # エピソードの最大ステップ数を1000に設定
        window_size = 50,  # 環境で使用するデータのウィンドウサイズを50に設定
        reward_function = AccountValueChangeReward(),  # 報酬関数としてアカウント価値変化報酬を使用
        metrics = [   # パフォーマンス指標のリスト
            DifferentActions(),  # 異なるアクションの数を測定
            AccountValue(),      # アカウントの価値を測定
            MaxDrawdown(),       # 最大ドローダウンを測定
            SharpeRatio(),       # シャープレシオを測定
        ]
    )
    # 環境からアクション空間と観察空間の形状を取得
    action_space = env.action_space  # 環境のアクション空間を取得
    input_shape = env.observation_space.shape  # 環境の観察空間の形状を取得

    """ アクター・モデルを定義する関数 """
    def actor_model(input_shape, action_space):
        # 入力層を作成
        input = layers.Input(shape=input_shape, dtype=tf.float32)  # 観察空間の形状を持つ入力層を定義
        x = layers.Flatten()(input)  # 入力を平坦化して1次元に変換

        # 隠れ層を追加
        x = layers.Dense(512, activation='elu')(x)  # 512ユニットの隠れ層(ELL活性化関数)
        x = layers.Dense(256, activation='elu')(x)  # 256ユニットの隠れ層（ELL活性化関数）
        x = layers.Dense(64, activation='elu')(x)   # 64ユニットの隠れ層（ELL活性化関数）
        x = layers.Dropout(0.2)(x)  # ドロップアウト層(20%のユニットをランダムに無効化)
        
        # 出力層を作成（離散アクション空間用）
        output = layers.Dense(action_space, activation='softmax')(x) # アクション空間のユニット数を持つ出力層（softmax活性化関数）

        # モデルを定義し、入力と出力を指定
        return models.Model(inputs=input, outputs=output)  # アクターモデルを返す

    """ クリティック・モデルを定義する関数 """
    def critic_model(input_shape):
        # 入力層を作成
        input = layers.Input(shape=input_shape, dtype=tf.float32)  # 観察空間の形状を持つ入力層を定義
        x = layers.Flatten()(input)  # 入力を平坦化して1次元に変換

        # 隠れ層を追加
        x = layers.Dense(512, activation='elu')(x)  # 512ユニットの隠れ層（ELU活性化関数）
        x = layers.Dense(256, activation='elu')(x)  # 256ユニットの隠れ層（ELU活性化関数）
        x = layers.Dense(64, activation='elu')(x)   # 64ユニットの隠れ層（ELU活性化関数）
        x = layers.Dropout(0.2)(x)  # ドロップアウト層（20%のユニットをランダムに無効化）

        # 出力層を作成
        output = layers.Dense(1, activation=None)(x)  # 1ユニットの出力層（活性化なし）
        
        # モデルを定義し、入力と出力を指定
        return models.Model(inputs=input, outputs=output)  # クリティックモデルを返す
        
    # PPOAgentのインスタンスを作成
    agent = PPOAgent(
        actor=actor_model(input_shape, action_space),  # アクターモデルを指定
        critic=critic_model(input_shape),  # クリティックモデルを指定
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Adamオプティマイザーを使用し、学習率を設定
        batch_size=128,  # バッチサイズを128に設定
        lamda=0.95,  # GAE（Generalized Advantage Estimation）のラムダ値
        kl_coeff=0.5,  # KLダイバージェンスの係数
        c2=0.01,  # クリティックの損失関数の係数
        writer_comment='ppo_sinusoid_discrete',  # ログに書き込むコメント
    )

    # エージェントのログディレクトリに設定を保存
    logdir = agent.logdir  # ログディレクトリを取得
    print(f"Log directory: {logdir}")
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    try:
        pd_data_feeder.save_config(logdir)  # PdDataFeederの設定をログディレクトリに保存
        env.env.save_config(logdir)  # 環境の設定をログディレクトリに保存
    except Exception as e:
        print(f"Failed to save config to file: {e}")  # エラーメッセージを表示


    # メモリマネージャーを作成し、環境の数を指定
    memory = MemoryManager(num_envs=num_envs)  # 複数環境の経験を管理するオブジェクト
    meanAverage = MeanAverage(best_mean_score_episode=1000)  # 平均スコアを管理するオブジェクト

    # 環境をリセットし、初期状態と情報を取得
    states, infos = env.reset()  
    rewards = 0.0  # 報酬を初期化

    while True:  # 無限ループでエピソードを実行
        action, prob = agent.act(states)  # エージェントが現在の状態に基づいてアクションを選択し、その確率も取得

        # 環境にアクションを適用し、次の状態、報酬、終了フラグ、トランケートフラグ、情報を取得
        next_states, reward, terminated, truncated, infos = env.step(action)  
        
        # メモリに現在の状態、アクション、報酬、確率、終了フラグ、トランケートフラグ、次の状態、情報を追加
        memory.append(states, action, reward, prob, terminated, truncated, next_states, infos)  
        
        # 現在の状態を次の状態に更新
        states = next_states  

        # メモリ内の完了したエピソードのインデックスを取得し、学習を行う
        for index in memory.done_indices():
            env_memory = memory[index]  # 完了したエピソードのメモリを取得
            history = agent.train(env_memory)  # エージェントをトレーニング
            mean_reward = meanAverage(np.sum(env_memory.rewards))  # 平均報酬を計算

            # 平均報酬がベストの場合、モデルを保存
            if meanAverage.is_best(agent.epoch):
                agent.save_models('ppo_sinusoid', include_optimizer=True)  # モデルを保存する際にオプティマイザを含める

            # KLダイバージェンスが閾値を超えた場合、学習率を減少
            if history['kl_div'] > 0.05 and agent.epoch > 1000:
                agent.reduce_learning_rate(0.995, verbose=False)  # 学習率を減少させる

            # 最後の情報を取得
            info = env_memory.infos[-1]  
            
            # エポック、累積報酬、平均報酬、アカウント価値、KLダイバージェンスを出力
            print(agent.epoch, np.sum(env_memory.rewards), mean_reward, info["metrics"]['account_value'], history['kl_div'])  
            
            # メトリックをログに書き込む
            agent.log_to_writer(info['metrics'])  
            
            # 環境をリセットし、新しい状態と情報を取得
            states[index], infos[index] = env.reset(index=index)  

        # エポックが10000に達したらループを終了
        if agent.epoch >= 10000:
            break  

    # 環境を閉じる
    env.close()  
    exit()  # プログラムを終了