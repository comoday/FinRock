import pstats

# プロファイル結果を読み込む
p = pstats.Stats('profile_results.prof')

# 結果を表示する（デフォルトで時間順にソート）
p.sort_stats('cumulative').print_stats(10)  # 上位10件を表示
