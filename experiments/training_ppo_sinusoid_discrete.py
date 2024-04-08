import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from keras import layers, models

from finrock.data_feeder import PdDataFeeder
from finrock.trading_env import TradingEnv, ActionSpace
from finrock.scalers import MinMaxScaler, ZScoreScaler, LogReturnsScaler, Normalizer
from finrock.reward import SimpleReward, AccountValueChangeReward
from finrock.metrics import DifferentActions, AccountValue, MaxDrawdown, SharpeRatio
from finrock.indicators import BolingerBands, RSI, PSAR, SMA, MACD

from rockrl.utils.misc import MeanAverage
from rockrl.utils.memory import MemoryManager
from rockrl.tensorflow import PPOAgent
from rockrl.utils.vectorizedEnv import VectorizedEnv

df = pd.read_csv('Datasets/random_sinusoid.csv')
df, df_test = df[:-1000], df[-1000:]


pd_data_feeder = PdDataFeeder(
    df,
    indicators = [
        BolingerBands(data=df, period=20, std=2),
        RSI(data=df, period=14),
        PSAR(data=df),
        MACD(data=df),
        SMA(data=df, period=7),
    ]
)

num_envs = 10
env = VectorizedEnv(
    env_object = TradingEnv,
    num_envs = num_envs,
    data_feeder = pd_data_feeder,
    output_transformer = Normalizer(),
    initial_balance = 1000.0,
    max_episode_steps = 1000,
    window_size = 100,
    reward_function = AccountValueChangeReward(),
    metrics = [
        DifferentActions(),
        AccountValue(),
        MaxDrawdown(),
        SharpeRatio(),
    ],
    # action_space = ActionSpace.SHORT_DISCRETE,
    action_space = ActionSpace.DISCRETE,
)

action_space = env.action_space
input_shape = env.observation_space.shape

# def actor_model(input_shape, action_space):
#     input = layers.Input(shape=input_shape, dtype=tf.float32)
#     x = layers.Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(input)
#     x = layers.MaxPooling1D(pool_size=2)(x)
#     x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(x)
#     x = layers.MaxPooling1D(pool_size=2)(x)
#     x = layers.Flatten()(x)
#     # x = layers.Flatten()(input)
#     # x = layers.Dense(512, activation='elu')(x)
#     # x = layers.Dense(256, activation='elu')(x)
#     x = layers.Dense(64, activation='elu')(x)
#     # x = layers.Dropout(0.5)(x)
#     output = layers.Dense(action_space, activation='softmax')(x) # discrete action space
#     return models.Model(inputs=input, outputs=output)

# def critic_model(input_shape):
#     input = layers.Input(shape=input_shape, dtype=tf.float32)
#     x = layers.Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(input)
#     x = layers.MaxPooling1D(pool_size=2)(x)
#     x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(x)
#     x = layers.MaxPooling1D(pool_size=2)(x)
#     x = layers.Flatten()(x)
#     # x = layers.Dense(512, activation='elu')(x)
#     # x = layers.Dense(256, activation='elu')(x)
#     x = layers.Dense(64, activation='elu')(x)
#     # x = layers.Dropout(0.5)(x)
#     output = layers.Dense(1, activation=None)(x)
#     return models.Model(inputs=input, outputs=output)



def actor_model(input_shape, action_space):
    input = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Flatten()(input)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(input)
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(64)(x)
    output = layers.Dense(action_space, activation='softmax')(x) # discrete action space
    return models.Model(inputs=input, outputs=output)

def critic_model(input_shape):
    input = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Flatten()(input)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(input)
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(64)(x)
    output = layers.Dense(1, activation=None)(x)
    return models.Model(inputs=input, outputs=output)

agent = PPOAgent(
    actor = actor_model(input_shape, action_space),
    critic = critic_model(input_shape),
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=0.0001),
    batch_size=512,
    lamda=0.90,
    kl_coeff=0.5,
    c2=0.01,
    writer_comment='ppo_sinusoid_discrete',
)
agent.summary()

pd_data_feeder.save_config(agent.logdir)
env.env.save_config(agent.logdir)

# environment for testing
test_env = TradingEnv.load_config(
    PdDataFeeder.load_config(df_test, agent.logdir), 
    agent.logdir
)


memory = MemoryManager(num_envs=num_envs)
meanAverage = MeanAverage(best_mean_score_episode=1000)
states, infos = env.reset()
rewards = 0.0
while True:
    action, prob = agent.act(states)

    next_states, reward, terminated, truncated, infos = env.step(action)
    memory.append(states, action, reward, prob, terminated, truncated, next_states, infos)
    states = next_states

    for index in memory.done_indices():
        env_memory = memory[index]
        history = agent.train(env_memory)
        mean_reward = meanAverage(np.sum(env_memory.rewards))

        if meanAverage.is_best(agent.epoch):
            agent.save_models('ppo_sinusoid')

        if history['kl_div'] > 0.10 and agent.epoch > 1000:
            agent.reduce_learning_rate(0.995, verbose=False)

        info = env_memory.infos[-1]
        print(agent.epoch, np.sum(env_memory.rewards), mean_reward, info["metrics"]['account_value'], history['kl_div'])
        agent.log_to_writer(info['metrics'])
        states[index], infos[index] = env.reset(index=index)

        # test after last environment is done
        if index == num_envs - 1:
            test_state, test_info = test_env.reset()
            while True:
                action, _ = agent.act(test_state, training=False)
                test_state, reward, terminated, truncated, info = test_env.step(action)

                if terminated or truncated:
                    for metric, value in info['metrics'].items():
                        agent.log_to_writer({f'test/{metric}': value})
                    break

    if agent.epoch >= 20000:
        break

env.close()
agent.close()