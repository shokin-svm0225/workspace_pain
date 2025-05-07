import pandas as pd

# CSVファイルのリストを指定
csv_files = ['null/peinditect/侵害受容性疼痛_filtered_data.csv',
             'null/peinditect/神経障害性疼痛_filtered_data.csv',
             'null/peinditect/不明_filtered_data.csv',
             '欠損値補完/PAINDITECT/det_painditect_KNN_侵害受容性疼痛.csv',
             '欠損値補完/PAINDITECT/det_painditect_KNN_神経障害性疼痛.csv',
             '欠損値補完/PAINDITECT/det_painditect_KNN_不明.csv',
             '欠損値補完/PAINDITECT/det_painditect_mean_侵害受容性疼痛.csv',
             '欠損値補完/PAINDITECT/det_painditect_mean_神経障害性疼痛.csv',
             '欠損値補完/PAINDITECT/det_painditect_mean_不明.csv',
             '欠損値補完/PAINDITECT/det_painditect_median_侵害受容性疼痛.csv',
             '欠損値補完/PAINDITECT/det_painditect_median_神経障害性疼痛.csv',
             '欠損値補完/PAINDITECT/det_painditect_median_不明.csv',]

# 各CSVファイルに対して処理を行う
for csv_file in csv_files:
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)
    
    # 例として、列Aに定数2をかける
    df['痛みの経過（表）'] = df['痛みの経過（表）'] * 1.5
    df['⑦'] = df['⑦'] * 1.5

    # 新しいCSVファイルとして保存する（元のファイル名に "_modified" を追加）
    output_csv_file_path = csv_file.replace('.csv', '_weighting.csv')
    df.to_csv(output_csv_file_path, index=False)