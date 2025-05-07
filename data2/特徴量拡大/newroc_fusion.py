import pandas as pd

# CSVファイルのリストを指定
csv_files = ['null/fusion/questionnaire_fusion_missing_侵害受容性疼痛.csv',
             'null/fusion/questionnaire_fusion_missing_神経障害性疼痛.csv',
             'null/fusion/questionnaire_fusion_missing_不明.csv',
             '欠損値補完/FUSION/det_KNN_侵害受容性疼痛.csv',
             '欠損値補完/FUSION/det_KNN_神経障害性疼痛.csv',
             '欠損値補完/FUSION/det_KNN_不明.csv',
             '欠損値補完/FUSION/det_mean_侵害受容性疼痛.csv',
             '欠損値補完/FUSION/det_mean_神経障害性疼痛.csv',
             '欠損値補完/FUSION/det_mean_不明.csv',
             '欠損値補完/FUSION/det_median_侵害受容性疼痛.csv',
             '欠損値補完/FUSION/det_median_神経障害性疼痛.csv',
             '欠損値補完/FUSION/det_median_不明.csv',]

# 列名を指定
column1 = '②'
column2 = '⑥'
new_column1 = '痺れ'
column3 = '③'
column4 = '⑦'
new_column2 = '少しの痛み'
column5 = '④.1'
column6 = '④.2'
new_column3 = '機嫌'
column7 = '⑥.1'
column8 = '⑧.1'
new_column4 = 'しつこさ'

# 各CSVファイルに対して処理を行う
for csv_file in csv_files:
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)

    # 列の掛け算をして新しい列を追加する
    df[new_column1] = df[column1] * df[column2]
    df[new_column2] = df[column3] * df[column4]
    df[new_column3] = df[column5] * df[column6]
    df[new_column4] = df[column7] * df[column8]

    # 新しいCSVファイルとして保存する（元のファイル名に "_modified" を追加）
    output_csv_file_path = csv_file.replace('.csv', '_newroc.csv')
    df.to_csv(output_csv_file_path, index=False)