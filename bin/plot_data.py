import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルからデータを読込み、タイムスタンプをインデックスとして設定し、日付を解析
df = pd.read_csv('Datasets/random_sinusoid.csv', index_col='timestamp', parse_dates=True)

# 'open','high','low','close'の列のみ選択
df = df[['open', 'high', 'low', 'close']]
# データポイントをラスト1000件に制限
df = df[-1000:]

# データのプロット
plt.figure(figsize=(10, 6))
plt.plot(df['close'])
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('random_sinusoid.csv')
plt.grid(True)
plt.show()