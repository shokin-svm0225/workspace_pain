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

# 各CSVファイルに対して処理を行う
for csv_file in csv_files:
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)
    
    # 例として、列Aに定数1をかける
    df['②'] = df['②'] * 0.5
    df['③'] = df['③'] * 0.5
    df['⑤'] = df['⑤'] * 0.5
    df['⑤.1'] = df['⑤.1'] * 0.5
    df['⑥.1'] = df['⑥.1'] * 0.5

    # 新しいCSVファイルとして保存する（元のファイル名に "_modified" を追加）
    output_csv_file_path = csv_file.replace('.csv', '_weighting.csv')
    df.to_csv(output_csv_file_path, index=False)