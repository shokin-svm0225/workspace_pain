import pandas as pd
import numpy as np

df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/データ/MedicalData_columns_change.csv')

#'#REF!'という値をnp.nan（欠損値）に置き換え,欠損値を含む行を削除
df1 = df1.replace('#REF!', np.nan).dropna()

#数値データ(浮動小数点型(float)および整数型(int))のみを抽出
number_data = df1.select_dtypes(include=[float, int])

#数値データを四捨五入して整数型に変換
imputed_data = np.round(number_data).astype(int)

#元のデータフレームの数値データ部分を更新
df1[number_data.columns] = imputed_data

#結果を指定の新しいCSVファイルに保存
df1.to_csv('questionnaire_fusion_missing.csv', index=False)
