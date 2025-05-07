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

# 各CSVファイルに対して処理を行う
for csv_file in csv_files:
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)
    
    # 例として、列Aに定数2をかける
    df['痛みの経過（表）'] = df['痛みの経過（表）'] * 1.5
    df['⑦'] = df['⑦'] * 1.5
    df['②.1'] = df['②.1'] * 0.5
    df['③.1'] = df['③.1'] * 0.5
    df['⑤.1'] = df['⑤.1'] * 0.5
    df['⑤.2'] = df['⑤.2'] * 0.5
    df['⑥.2'] = df['⑥.2'] * 0.5

    # 新しいCSVファイルとして保存する（元のファイル名に "_modified" を追加）
    output_csv_file_path = csv_file.replace('.csv', '_weighting.csv')
    df.to_csv(output_csv_file_path, index=False)