import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import numpy as np

df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/MedicalData_columns_change.csv')

df1.replace(['#REF!', 'N/A', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)

number_data = df1.select_dtypes(include=[float, int])

class KNN:
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(number_data)
    imputed_data = np.round(imputed_data).astype(int)

    df1[number_data.columns] = imputed_data
    df1.fillna(0, inplace=True)

    det_other = df1[~df1['痛みの種類'].str.match(r'^侵害受容性疼痛$|^神経障害性疼痛$', na=False)]
    det_inj_pain = df1[df1['痛みの種類'].str.match(r'^侵害受容性疼痛$', na=False)]
    det_neuro_pain = df1[df1['痛みの種類'].str.match(r'^神経障害性疼痛$', na=False)]

    det_other.to_csv('det_KNN_不明.csv', index=False)
    det_inj_pain.to_csv('det_KNN_侵害受容性疼痛.csv', index=False)
    det_neuro_pain.to_csv('det_KNN_神経障害性疼痛.csv', index=False)

class median:
    imputer = SimpleImputer(strategy='median')

    imputed_data = imputer.fit_transform(number_data)

    imputed_data = np.round(imputed_data).astype(int)

    df1[number_data.columns] = imputed_data
    df1.fillna(0, inplace=True)

    det_other = df1[~df1['痛みの種類'].str.match(r'^侵害受容性疼痛$|^神経障害性疼痛$', na=False)]
    det_inj_pain = df1[df1['痛みの種類'].str.match(r'^侵害受容性疼痛$', na=False)]
    det_neuro_pain = df1[df1['痛みの種類'].str.match(r'^神経障害性疼痛$', na=False)]

    det_other.to_csv('det_median_不明.csv', index=False)
    det_inj_pain.to_csv('det_median_侵害受容性疼痛.csv', index=False)
    det_neuro_pain.to_csv('det_median_神経障害性疼痛.csv', index=False)

class mean:
    imputer = SimpleImputer(strategy='mean')

    imputed_data = imputer.fit_transform(number_data)

    imputed_data = np.round(imputed_data).astype(int)

    df1[number_data.columns] = imputed_data
    df1.fillna(0, inplace=True)

    det_other = df1[~df1['痛みの種類'].str.match(r'^侵害受容性疼痛$|^神経障害性疼痛$', na=False)]
    det_inj_pain = df1[df1['痛みの種類'].str.match(r'^侵害受容性疼痛$', na=False)]
    det_neuro_pain = df1[df1['痛みの種類'].str.match(r'^神経障害性疼痛$', na=False)]

    det_other.to_csv('det_mean_不明.csv', index=False)
    det_inj_pain.to_csv('det_mean_侵害受容性疼痛.csv', index=False)
    det_neuro_pain.to_csv('det_mean_神経障害性疼痛.csv', index=False)

KNN()
median()
mean()
