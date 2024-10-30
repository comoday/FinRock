import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

""" サイン波データを持つデータフレームを作成 """
def create_sinusoidal_df(
    amplitude = 2000.0,           # 価格の変動の振幅
    frequency = 0.01,             # 価格の変動の周波数
    phase = 0.0,                  # 価格の変動の位相シフト
    num_samples = 10000,          # データサンプルの数
    data_shift = 20000,           # データを上にシフト
    trendline_down = 5000,        # データを下にシフト
    plot = False,                 # プロットを表示するかどうか
    ):
    

    # 時間軸を生成
    t = np.linspace(0, 2 * np.pi * frequency * num_samples, num_samples)

    # 現在の日時を取得
    now = datetime.now()

    # 時間、分、秒をゼロに設定
    now = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # 各日付のタイムスタンプを生成
    # timestamps = [now - timedelta(days=i) for i in range(num_samples)]
    timestamps = [now - timedelta(hours=i*4) for i in range(num_samples)]

    # datetimeオブジェクトを文字列に変換
    timestamps = [timestamps.strftime('%Y-%m-%d %H:%M:%S') for timestamps in timestamps]

    # タイムスタンプの順序を反転
    timestamps = timestamps[::-1]

    # 価格のためのサインはデータを生成
    sin_data = amplitude * np.sin(t + phase)   # サイン波の生成
    sin_data += data_shift                     # データを上にシフト

    # トレンドラインを作成するためにデータを下にシフト
    sin_data -= np.linspace(0, trendline_down, num_samples)

    # ランダムノイズを追加
    noise = np.random.uniform(0.95, 1.05, len(t))  # ランダムノイズを生成
    noisy_sin_data = sin_data * noise  # 元データにノイズを追加

    # ノイズ付きサインはデータの価格範囲を計算
    price_range = np.max(noisy_sin_data) - np.min(noisy_sin_data)

    # ランダムな低価格と終値を生成
    low_prices = noisy_sin_data - np.random.uniform(0, 0.1 * price_range, len(noisy_sin_data))   # 低価格はノイズデータからランダムに引く
    close_prices = noisy_sin_data + np.random.uniform(-0.05 * price_range, 0.05 * price_range, len(noisy_sin_data))   # 終値はノイズデータにランダムに加減

    # 始値は通常、前日の終値に近い
    open_prices = np.zeros(len(close_prices))  # 始値の配列を初期化
    open_prices[0] = close_prices[0]           # 最初の始値を最初の終値に設定
    open_prices[1:] = close_prices[:-1]        # 以降の始値を前日の終値に設定

    # 高値は常に始値と終値の上にある
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 0.1 * price_range, len(close_prices))   # 高値を計算

    # 低値は常に始値と終値の下にある
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 0.1 * price_range, len(close_prices))   # 低値を計算

    # プロットを表示する場合
    if plot:
        # 価格データをプロット
        plt.figure(figsize=(10, 6))
        plt.plot(t, noisy_sin_data, label='Noisy Sinusoidal Data')  # ノイズ付きサイン波データ
        plt.plot(t, open_prices, label='Open')
        plt.plot(t, low_prices, label='Low')
        plt.plot(t, close_prices, label='Close')
        plt.plot(t, high_prices, label='High')
        plt.xlabel('Time')   # ｘ軸ラベル
        plt.ylabel('Price')  # ｙ軸ラベル
        plt.title('Fake Price Data')   # グラフタイトル['Fake Price Data']
        plt.legend()   # 凡例を表示
        plt.grid(True)   # グリッド線を表示
        plt.show()

    # データフレームdf[['open', 'high', 'low', 'close']の作成
    df = pd.DataFrame({'timestamp': timestamps, 'open': open_prices, 'high': high_prices, 'low': low_prices, 'close': close_prices})

    return df   # データフレームを返す

if __name__ == '__main__':
    # サイン波データを持つデータフレームを作成
    df = create_sinusoidal_df()

    # データセットを保存するディレクトリを作成
    os.makedirs('Datasets', exist_ok=True)

    # データフレームをCSVファイルに保存
    df.to_csv(f'Datasets/random_sinusoid.csv')