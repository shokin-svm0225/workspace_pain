import pandas as pd
import os

# CSVファイルが格納されているフォルダのパスを指定してください
folder_path = '/Users/iwasho_0225/Desktop/workspace/pain_experiment/null/コピー'

# フォルダ内のすべてのCSVファイルを処理
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        # CSVファイルを読み込み
        df = pd.read_csv(file_path)
        
        # 3列目から10列目を削除（インデックスは0から始まるので、2から9に対応）
        df.drop(df.columns[15], axis=1, inplace=True, errors='ignore')

        #小数点が出ないように整数として扱える列を確認して変換
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]) and (df[col] % 1 == 0).all():
                df[col] = df[col].astype(int)

        # ファイルを上書き保存 (float_format="%.0f" を使用して小数点が出ないようにする)
        df.to_csv(file_path, index=False, float_format="%.0f")
        
        print(f"{filename} の3列目から10列目を削除して保存しました。")
