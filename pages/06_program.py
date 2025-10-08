import streamlit as st
import itertools
import plotly.express as px
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
from streamlit_option_menu import option_menu

st.title('プログラム')

# ラジオボタンを表示
home_type = st.sidebar.radio("選んでください", ["欠損値補完", "特徴量拡大", "SVM(cv2)", "SVM(scikit-learn)", "標準化", "交差検証", "感度と特異度の計算"])

if home_type == "欠損値補完":
    st.subheader('欠損値補完', divider='rainbow')
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

if home_type == "特徴量拡大":
    st.subheader('特徴量拡大', divider='rainbow')
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

if home_type == "SVM(cv2)":
    st.subheader('SVM(cv2)', divider='rainbow')
    body_3 = """
    import cv2

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
    
    svm = cv2.ml.SVM_load(SAVE_TRAINED_DATA_PATH)
    _, predicted = svm.predict(test_datas)
    """
    st.code(body_3, language="python")

if home_type == "SVM(scikit-learn)":
    st.subheader('SVM(scikit-learn)', divider='rainbow')
    body_4 = """
    from sklearn.svm import SVC
    import joblib

    svm = SVC(C=C, kernel='linear', max_iter=1500)
    svm.fit(X_train, y_train)# トレーニング

    # バリデーションデータで評価
    predicted = svm.predict(X_val)
    score = np.mean(y_val == predicted)
    scores.append(score)

    # モデル保存
    joblib.dump(best_model, MODEL_PATH)
    """
    st.code(body_4, language="python")

if home_type == "標準化":
    st.subheader('標準化', divider='rainbow')
    body_5 = """
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        datas = scaler.fit_transform(datas)
            """
    st.code(body_5, language="python")

if home_type == "交差検証":
    st.subheader('交差検証', divider='rainbow')
    body_6 = """
    from sklearn.model_selection import StratifiedKFold

    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)

    for train_index, val_index in skf.split(datas, labels):

        X_train, X_val = datas[train_index], datas[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        svm = SVC(C=C, kernel='linear', max_iter=1500)
        svm.fit(X_train, y_train)# トレーニング

        # バリデーションデータで評価
        predicted = svm.predict(X_val)
        score = np.mean(y_val == predicted)
        scores.append(score)

    avg_score = np.mean(scores)
    st.write(f"C: {C}, Score: {avg_score:.4f}")
    """
    st.code(body_6, language="python")

if home_type == "感度と特異度の計算":
    st.subheader('感度と特異度の計算', divider='rainbow')
    body_7 = """
    from sklearn.metrics import confusion_matrix

    conf_matrix = confusion_matrix(y_val, predicted, labels=[1, 2, 3])

    sensitivity_list = []
    specificity_list = []

    n_classes = conf_matrix.shape[0]
    
    for i in range(n_classes):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FN + FP)
        
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

        st.write(f"疼痛 {i+1}: 感度 = {sensitivity * 100:.2f}%, 特異度 = {specificity * 100:.2f}%")
    """
    st.code(body_7, language="python")