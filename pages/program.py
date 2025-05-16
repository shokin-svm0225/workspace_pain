import streamlit as st
import itertools
import plotly.express as px
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import cv2
import csv
import datetime
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu

def show():
    st.title('データ加工・機械学習のプログラムを表示')
    st.header('欠損値補完')
    body_1 = """
    import pandas as pd
    from sklearn.impute import KNNImputer
    from sklearn.impute import SimpleImputer
    import numpy as np

    df1 = pd.read_csv('null/peinditect/PAINDITECT.csv')

    #特定の文字列を欠損値に置き換え
    df1.replace(['#REF!', 'N/A', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)

    #数値データのみを抽出
    number_data = df1.select_dtypes(include=[float, int])

    class KNN:
        #最近傍法（K-Nearest Neighbors）で欠損値を5つの近傍を使用して補完
        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(number_data)
        imputed_data = np.round(imputed_data).astype(int)

        df1[number_data.columns] = imputed_data
        #求められた値が0の時反映されない場合があるため、残っている欠損値をすべて0で埋める
        df1.fillna(0, inplace=True)

        det_other = df1[~df1['痛みの種類'].str.match(r'^侵害受容性疼痛$|^神経障害性疼痛$', na=False)]
        det_inj_pain = df1[df1['痛みの種類'].str.match(r'^侵害受容性疼痛$', na=False)]
        det_neuro_pain = df1[df1['痛みの種類'].str.match(r'^神経障害性疼痛$', na=False)]

        det_other.to_csv('det_painditect_KNN_不明.csv', index=False)
        det_inj_pain.to_csv('det_painditect_KNN_侵害受容性疼痛.csv', index=False)
        det_neuro_pain.to_csv('det_painditect_KNN_神経障害性疼痛.csv', index=False)

    #中央値で欠損値を補完
    class median:
        imputer = SimpleImputer(strategy='median')

        imputed_data = imputer.fit_transform(number_data)

        imputed_data = np.round(imputed_data).astype(int)
        df1[number_data.columns] = imputed_data
        df1.fillna(0, inplace=True)

        det_other = df1[~df1['痛みの種類'].str.match(r'^侵害受容性疼痛$|^神経障害性疼痛$', na=False)]
        det_inj_pain = df1[df1['痛みの種類'].str.match(r'^侵害受容性疼痛$', na=False)]
        det_neuro_pain = df1[df1['痛みの種類'].str.match(r'^神経障害性疼痛$', na=False)]

        det_other.to_csv('det_painditect_median_不明.csv', index=False)
        det_inj_pain.to_csv('det_painditect_median_侵害受容性疼痛.csv', index=False)
        det_neuro_pain.to_csv('det_painditect_median_神経障害性疼痛.csv', index=False)

    #平均値で欠損値を補完
    class mean:
        imputer = SimpleImputer(strategy='mean')

        imputed_data = imputer.fit_transform(number_data)

        imputed_data = np.round(imputed_data).astype(int)

        df1[number_data.columns] = imputed_data
        df1.fillna(0, inplace=True)

        det_other = df1[~df1['痛みの種類'].str.match(r'^侵害受容性疼痛$|^神経障害性疼痛$', na=False)]
        det_inj_pain = df1[df1['痛みの種類'].str.match(r'^侵害受容性疼痛$', na=False)]
        det_neuro_pain = df1[df1['痛みの種類'].str.match(r'^神経障害性疼痛$', na=False)]

        det_other.to_csv('det_painditect_mean_不明.csv', index=False)
        det_inj_pain.to_csv('det_painditect_mean_侵害受容性疼痛.csv', index=False)
        det_neuro_pain.to_csv('det_painditect_mean_神経障害性疼痛.csv', index=False)

    KNN()
    median()
    mean()
    """
    st.code(body_1, language="python")

    st.header('特徴量増量')
    body_2 = """
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
    """
    st.code(body_2, language="python")
    
    st.header('svm実装')
    body_3 = """
    TEST_DATA_RATIO = 0.3
    SAVE_TRAINED_DATA_PATH = "svm_data.xml"

    # csvファイルの読み込み
    df = pd.read_csv(uploaded_file, encoding = 'utf-8')

    # カラムと重みの値を取得
    columns = df["columns"].tolist()
    weights = df["weights"].tolist()
    
    # データの分割
    df_nociceptive_train, df_nociceptive_test = train_test_split(
        df1[columns], test_size=TEST_DATA_RATIO, random_state=None
        )
    df_neuronociceptive_train, df_neuronociceptive_test = train_test_split(
        df2[columns], test_size=TEST_DATA_RATIO, random_state=None
        )
    df_unknown_train, df_unknown_test = train_test_split(
        df3[columns], test_size=TEST_DATA_RATIO, random_state=None
        )

    # 重みを適用して特徴量を調整（訓練データの場合）
    df_nociceptive_train_weighted = df_nociceptive_train.mul(weights, axis=1)
    df_nociceptive_test_weighted = df_nociceptive_test.mul(weights, axis=1)

    # トレーニングデータとラベルの作成
    datas = np.vstack(
        [
            df_nociceptive_train.values,
            df_neuronociceptive_train.values,
            df_unknown_train.values,
            ]
            ).astype(np.float32)
    
    labels1 = np.full(len(df_nociceptive_train), 1, np.int32)
    labels2 = np.full(len(df_neuronociceptive_train), 2, np.int32)
    labels3 = np.full(len(df_unknown_train), 3, np.int32)
    labels = np.concatenate([labels1, labels2, labels3]).astype(np.int32)
    
    # SVMモデルの作成とトレーニング
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setGamma(1)
    svm.setC(1)
    svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1500, 1.0e-06))
    svm.train(datas, cv2.ml.ROW_SAMPLE, labels)
    
    # モデルを保存
    svm.save(SAVE_TRAINED_DATA_PATH)
    
    test_datas = np.vstack(
        [
        df_nociceptive_test.values,
        df_neuronociceptive_test.values,
        df_unknown_test.values,
        ]
        ).astype(np.float32)
    
    test_labels1 = np.full(len(df_nociceptive_test), 1, np.int32)
    test_labels2 = np.full(len(df_neuronociceptive_test), 2, np.int32)
    test_labels3 = np.full(len(df_unknown_test), 3, np.int32)
    
    test_labels = np.concatenate([test_labels1, test_labels2, test_labels3]).astype(
        np.int32
        )
    
    # # データの標準化
    # scaler = StandardScaler()
    # datas = scaler.fit_transform(datas)
    # test_datas = scaler.transform(test_datas)
    
    # # 交差検証の実行
    # cross_val_scores = cross_val_score(svm, datas, labels, cv=5)
    # print("Cross-Validation Scores:", cross_val_scores)
    # print("Mean Cross-Validation Score:", cross_val_scores.mean())
    
    svm = cv2.ml.SVM_load(SAVE_TRAINED_DATA_PATH)
    _, predicted = svm.predict(test_datas)
    
    confusion_matrix = np.zeros((3, 3), dtype=int)
    
    for i in range(len(test_labels)):
        index1 = test_labels[i] - 1
        index2 = predicted[i][0] - 1
        confusion_matrix[int(index1)][int(index2)] += 1
        
    st.write("confusion matrix")
    st.table(confusion_matrix)

    score = np.sum(test_labels == predicted.flatten()) / len(test_labels)
        
    st.write("正答率:", score*100, "%")
        
    # 感度と特異度の計算
    sensitivity = np.zeros(3)
    specificity = np.zeros(3)
    
    for i in range(3):
        TP = confusion_matrix[i, i]
        FN = np.sum(confusion_matrix[i, :]) - TP
        FP = np.sum(confusion_matrix[:, i]) - TP
        TN = np.sum(confusion_matrix) - (TP + FN + FP)
        
        sensitivity[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity[i] = TN / (TN + FP) if (TN + FP) != 0 else 0
        
    # 感度と特異度の表示
    st.write("感度と特異度")
    st.write("（疼痛1:侵害受容性疼痛,疼痛2:神経障害性疼痛,疼痛3:不明）")
    for i in range(3):
        st.write(f"疼痛 {i+1}: 感度 = {sensitivity[i]:.4f}, 特異度 = {specificity[i]:.4f}")
    """
    st.code(body_3, language="python")