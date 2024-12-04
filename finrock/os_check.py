import os

# app_env = os.environ.get("APP_ENV")
# print(app_env)

# 環境変数を設定（追加・上書き）
os.environ['bybit_apiKey'] = 'wIrfPGfi7ydTEHKTnj'
os.environ['bybit_secret'] = 'BWiHLHGhnpXoHDu88eCtaNFWFqJWXBFvjL2N'

# 環境変数の確認
print(os.environ.get('bybit_apiKey'))
print(os.environ.get('bybit_secret'))