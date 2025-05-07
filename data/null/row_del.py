import pandas as pd

# 対象のCSVファイルのパスを指定してください
file_path = '/Users/iwasho_0225/Desktop/workspace/pain_experiment/null/fusion/questionnaire_fusion_missingのコピー.csv'

# CSVファイルを読み込み
df = pd.read_csv(file_path)

# 「痛みの種類」カラムが「侵害受容性疼痛」でない行を削除
df = df[~df["痛みの種類"].isin(["侵害受容性疼痛", "神経障害性疼痛"])]

# ファイルを上書き保存
df.to_csv(file_path, index=False)

print(f"{file_path} の「痛みの種類」が「侵害受容性疼痛」でない行を削除して保存しました。")