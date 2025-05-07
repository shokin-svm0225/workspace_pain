import pandas as pd

# CSVファイルのリストを指定
csv_files = ['null/BSPOP/questionnaire_bspop_missing_侵害受容性疼痛.csv',
             'null/BSPOP/questionnaire_bspop_missing_精神障害性疼痛.csv',
             'null/BSPOP/questionnaire_bspop_missing_不明.csv',
             '欠損値補完/BSPOP/det_bspop_KNN_侵害受容性疼痛.csv',
             '欠損値補完/BSPOP/det_bspop_KNN_神経障害性疼痛.csv',
             '欠損値補完/BSPOP/det_bspop_KNN_不明.csv',
             '欠損値補完/BSPOP/det_bspop_mean_侵害受容性疼痛.csv',
             '欠損値補完/BSPOP/det_bspop_mean_神経障害性疼痛.csv',
             '欠損値補完/BSPOP/det_bspop_mean_不明.csv',
             '欠損値補完/BSPOP/det_bspop_median_侵害受容性疼痛.csv',
             '欠損値補完/BSPOP/det_bspop_median_神経障害性疼痛.csv',
             '欠損値補完/BSPOP/det_bspop_median_不明.csv',]

# 列名を指定
column1 = '④'
column2 = '④.1'
new_column1 = '機嫌'
column3 = '⑥'
column4 = '⑧'
new_column2 = 'しつこさ'

# 各CSVファイルに対して処理を行う
for csv_file in csv_files:
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)

    # 列の掛け算をして新しい列を追加する
    df[new_column1] = df[column1] * df[column2]
    df[new_column2] = df[column3] * df[column4]

    # 新しいCSVファイルとして保存する（元のファイル名に "_modified" を追加）
    output_csv_file_path = csv_file.replace('.csv', '_newroc.csv')
    df.to_csv(output_csv_file_path, index=False)