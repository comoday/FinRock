# ファイル名: profile_indicators.py

import cProfile
import pandas as pd
from finrock.indicators import SMA, MACD, RSI, BolingerBands, PSAR

def compute_indicators(data):
    # インジケータのインスタンスを作成
    sma = SMA(data)
    macd = MACD(data)
    rsi = RSI(data)
    bolinger = BolingerBands(data)
    psar = PSAR(data)

    # 各インジケータの計算を実行
    sma.compute()
    macd.compute()
    rsi.compute()
    bolinger.compute()
    psar.compute()

def main():
    # サンプルデータの作成
    data = pd.DataFrame({
    'high': [81000.0, 81050.0, 81100.0, 81200.0, 80950.0] * 1000,  # 'high' カラム
    'low': [80000.0, 80050.0, 80500.0, 80550.0, 80400.0] * 1000,   # 'low' カラム
    'close': [80377.6, 80377.6, 80908.5, 81600.0, 80706.1] * 1000   # 'close' カラム
    })
    
    # インジケータの計算をプロファイリング
    profiler = cProfile.Profile()
    profiler.enable()
    
    compute_indicators(data)  # 直接呼び出し
    
    profiler.disable()
    profiler.dump_stats('profile_results.prof')  # 結果をファイルに保存


if __name__ == "__main__":
    main()