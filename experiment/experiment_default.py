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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu
from sklearn.svm import SVC
import joblib
import time

TEST_DATA_RATIO = 0.3
MODEL_PATH = "svm_model.pkl"

def parse_num_list(s, dtype=float):
    """
    '0.1, 1, 10' → [0.1, 1.0, 10.0]
    空文字や不正値は無視します。dtype は float / int を想定。
    """
    if not s:
        return []
    out = []
    for chunk in s.replace("，", ",").split(","):
        chunk = chunk.strip()
        if chunk == "":
            continue
        try:
            out.append(dtype(chunk))
        except Exception:
            pass
    return out

def cv_score_for_params(datas, labels, C, kernel, gamma, degree, coef0, k, max_iter_svc, seed=None):
    """与えられたパラメータで StratifiedKFold の平均正解率を返す。"""
    kwargs = dict(C=C, kernel=kernel, max_iter=max_iter_svc)
    if kernel == "linear":
        kwargs.update(gamma="scale", degree=3, coef0=0.0)
    elif kernel == "rbf":
        kwargs.update(gamma=gamma, degree=3, coef0=0.0)
    elif kernel == "poly":
        kwargs.update(gamma=gamma, degree=degree, coef0=coef0)
    elif kernel == "sigmoid":
        kwargs.update(gamma=gamma, degree=3, coef0=coef0)


    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, va_idx in skf.split(datas, labels):
        X_tr, X_va = datas[tr_idx], datas[va_idx]
        y_tr, y_va = labels[tr_idx], labels[va_idx]
        model = SVC(**kwargs)
        model.fit(X_tr, y_tr)
        scores.append(model.score(X_va, y_va))

    avg = float(np.mean(scores))
    return avg, dict(C=C, kernel=kernel, gamma=kwargs["gamma"], degree=kwargs["degree"], coef0=kwargs["coef0"])


# =========================
# UI & メイン処理
# =========================
def default_experiment():
    st.sidebar.header("最適化の設定")
    kernel = st.sidebar.selectbox("SVMカーネル", ["rbf", "linear", "poly", "sigmoid"], index=0, help="使用するカーネル")
    k_cv = st.sidebar.slider("StratifiedKFold の分割数 (k)", min_value=2, max_value=8, value=5, step=1)
    max_iter_svc = st.sidebar.number_input("SVC の max_iter", min_value=-1, max_value=50000, value=1500, step=100)

    st.sidebar.header("グリッドサーチ値")
    C_values = parse_num_list(st.sidebar.text_input("C", value="0.1, 1, 10"), float)

    # カーネル別パラメータ
    gamma_values = []
    degree_values = []
    coef0_values = []

    if kernel in ["rbf", "poly", "sigmoid"]:
        gamma_values = parse_num_list(st.sidebar.text_input("gamma", value="0.01, 0.05"), float)
    if kernel == "poly":
        degree_values = parse_num_list(st.sidebar.text_input("degree (poly)", value="2, 3"), int)
    if kernel in ["poly", "sigmoid"]:
        coef0_values = parse_num_list(st.sidebar.text_input("coef0 (poly/sigmoid)", value="0.0, 0.5"), float)

    # デフォルト補正（空回避）
    if not C_values:
        C_values = [1.0]
    if kernel in ["rbf", "poly", "sigmoid"] and not gamma_values:
        gamma_values = [0.01]
    if kernel == "poly" and not degree_values:
        degree_values = [3]
    if kernel in ["poly", "sigmoid"] and not coef0_values:
        coef0_values = [0.0]

    # セレクトボックスのオプションを定義
    options = ['欠損値データ削除', '中央値補完', '平均値補完', 'k-NN法補完']

    # セレクトボックスを作成し、ユーザーの選択を取得
    choice_1 = st.sidebar.selectbox('欠損値の対応', options, index = None, placeholder="選択してください")

    # セレクトボックスのオプションを定義
    options = ['PainDITECT', 'BS-POP', 'FUSION']

    # セレクトボックスを作成し、ユーザーの選択を取得
    choice_2 = st.sidebar.selectbox('使用する質問表', options, index = None, placeholder="選択してください")

    # ==== データ読込（元ロジックを踏襲） ====
    if choice_1 == '欠損値データ削除' and choice_2 == 'PainDITECT':
        df1 = pd.read_csv('data/null/peindetect/questionnaire_paindetect_missing_侵害.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/null/peindetect/questionnaire_paindetect_missing_神経.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/null/peindetect/questionnaire_paindetect_missing_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛'); st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛'); st.dataframe(df2)
        st.markdown('#### 原因不明'); st.dataframe(df3)

    elif choice_1 == '欠損値データ削除' and choice_2 == 'BS-POP':
        df1 = pd.read_csv('data/null/BSPOP/questionnaire_bspop_missing_侵害.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/null/BSPOP/questionnaire_bspop_missing_神経.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/null/BSPOP/questionnaire_bspop_missing_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛'); st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛'); st.dataframe(df2)
        st.markdown('#### 原因不明'); st.dataframe(df3)

    elif choice_1 == '欠損値データ削除' and choice_2 == 'FUSION':
        df1 = pd.read_csv('data/null/fusion/questionnaire_fusion_missing_侵害.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/null/fusion/questionnaire_fusion_missing_神経.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/null/fusion/questionnaire_fusion_missing_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛'); st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛'); st.dataframe(df2)
        st.markdown('#### 原因不明'); st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'PainDITECT':
        df1 = pd.read_csv('data/欠損値補完/PAINDITECT/det_median_侵害受容性疼痛_paindetect.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/PAINDITECT/det_median_神経障害性疼痛_paindetect.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/PAINDITECT/det_median_不明_paindetect.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛'); st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛'); st.dataframe(df2)
        st.markdown('#### 原因不明'); st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'BS-POP':
        df1 = pd.read_csv('data/欠損値補完/BSPOP/det_median_侵害受容性疼痛_bspop.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/BSPOP/det_median_神経障害性疼痛_bspop.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/BSPOP/det_median_不明_bspop.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛'); st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛'); st.dataframe(df2)
        st.markdown('#### 原因不明'); st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'FUSION':
        df1 = pd.read_csv('data/欠損値補完/FUSION/det_median_侵害受容性疼痛.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/FUSION/det_median_神経障害性疼痛.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/FUSION/det_median_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛'); st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛'); st.dataframe(df2)
        st.markdown('#### 原因不明'); st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'PainDITECT':
        df1 = pd.read_csv('data/欠損値補完/PAINDITECT/det_mean_侵害受容性疼痛_paindetect.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/PAINDITECT/det_mean_神経障害性疼痛_paindetect.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/PAINDITECT/det_mean_不明_paindetect.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛'); st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛'); st.dataframe(df2)
        st.markdown('#### 原因不明'); st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'BS-POP':
        df1 = pd.read_csv('data/欠損値補完/BSPOP/det_mean_侵害受容性疼痛_bspop.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/BSPOP/det_mean_神経障害性疼痛_bspop.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/BSPOP/det_mean_不明_bspop.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛'); st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛'); st.dataframe(df2)
        st.markdown('#### 原因不明'); st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'FUSION':
        df1 = pd.read_csv('data/欠損値補完/FUSION/det_mean_侵害受容性疼痛.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/FUSION/det_mean_神経障害性疼痛.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/FUSION/det_mean_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛'); st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛'); st.dataframe(df2)
        st.markdown('#### 原因不明'); st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'PainDITECT':
        df1 = pd.read_csv('data/欠損値補完/PAINDITECT/det_KNN_侵害受容性疼痛_paindetect.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/PAINDITECT/det_KNN_神経障害性疼痛_paindetect.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/PAINDITECT/det_KNN_不明_paindetect.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛'); st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛'); st.dataframe(df2)
        st.markdown('#### 原因不明'); st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'BS-POP':
        df1 = pd.read_csv('data/欠損値補完/BSPOP/det_KNN_侵害受容性疼痛_bspop.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/BSPOP/det_KNN_神経障害性疼痛_bspop.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/BSPOP/det_KNN_不明_bspop.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛'); st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛'); st.dataframe(df2)
        st.markdown('#### 原因不明'); st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'FUSION':
        df1 = pd.read_csv('data/欠損値補完/FUSION/det_KNN_侵害受容性疼痛.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/FUSION/det_KNN_神経障害性疼痛.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/FUSION/det_KNN_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛'); st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛'); st.dataframe(df2)
        st.markdown('#### 原因不明'); st.dataframe(df3)


    # 初期化
    if 'checkbox_states_1' not in st.session_state:
        st.session_state.checkbox_states_1 = {
            f"P{i}": False for i in range(1, 14)  # P1からP7まで初期化
        }

    # painditect が選ばれたときだけ、メイン画面に表示
    if choice_2 in ["PainDITECT", "FUSION"]:
        st.header("使用するカラムの指定(PainDITECT)")

        # 全選択・全解除ボタン
        col_buttons = st.columns(2)
        if col_buttons[0].button('全選択', key='select_all_1'):
            for key in st.session_state.checkbox_states_1:
                st.session_state.checkbox_states_1[key] = True
                st.session_state[key] = True
            st.rerun()

        if col_buttons[1].button('全解除', key='deselect_all_1'):
            for key in st.session_state.checkbox_states_1:
                st.session_state.checkbox_states_1[key] = False
                st.session_state[key] = False
            st.rerun()

        # チェックボックスの表示（元のスタイルを維持）
        col_1 = st.columns(7)
        painDITECT_1 = col_1[0].checkbox(label='P1', value=st.session_state.checkbox_states_1["P1"], key="P1")
        painDITECT_2 = col_1[1].checkbox(label='P2', value=st.session_state.checkbox_states_1["P2"], key="P2")
        painDITECT_3 = col_1[2].checkbox(label='P3', value=st.session_state.checkbox_states_1["P3"], key="P3")
        painDITECT_4 = col_1[3].checkbox(label='P4', value=st.session_state.checkbox_states_1["P4"], key="P4")
        painDITECT_5 = col_1[4].checkbox(label='P5', value=st.session_state.checkbox_states_1["P5"], key="P5")
        painDITECT_6 = col_1[5].checkbox(label='P6', value=st.session_state.checkbox_states_1["P6"], key="P6")
        painDITECT_7 = col_1[6].checkbox(label='P7', value=st.session_state.checkbox_states_1["P7"], key="P7")

        col_2 = st.columns(6)
        painDITECT_8 = col_2[0].checkbox(label='P8', value=st.session_state.checkbox_states_1["P8"], key="P8")
        painDITECT_9 = col_2[1].checkbox(label='P9', value=st.session_state.checkbox_states_1["P9"], key="P9")
        painDITECT_10 = col_2[2].checkbox(label='P10', value=st.session_state.checkbox_states_1["P10"], key="P10")
        painDITECT_11 = col_2[3].checkbox(label='P11', value=st.session_state.checkbox_states_1["P11"], key="P11")
        painDITECT_12 = col_2[4].checkbox(label='P12', value=st.session_state.checkbox_states_1["P12"], key="P12")
        painDITECT_13 = col_2[5].checkbox(label='P13', value=st.session_state.checkbox_states_1["P13"], key="P13")

        # 状態を反映
        st.session_state.checkbox_states_1["P1"] = painDITECT_1
        st.session_state.checkbox_states_1["P2"] = painDITECT_2
        st.session_state.checkbox_states_1["P3"] = painDITECT_3
        st.session_state.checkbox_states_1["P4"] = painDITECT_4
        st.session_state.checkbox_states_1["P5"] = painDITECT_5
        st.session_state.checkbox_states_1["P6"] = painDITECT_6
        st.session_state.checkbox_states_1["P7"] = painDITECT_7
        st.session_state.checkbox_states_1["P8"] = painDITECT_8
        st.session_state.checkbox_states_1["P9"] = painDITECT_9
        st.session_state.checkbox_states_1["P10"] = painDITECT_10
        st.session_state.checkbox_states_1["P11"] = painDITECT_11
        st.session_state.checkbox_states_1["P12"] = painDITECT_12
        st.session_state.checkbox_states_1["P13"] = painDITECT_13

    # 初期化
    if 'checkbox_states_2' not in st.session_state:
        st.session_state.checkbox_states_2 = {
            f"D{i}": False for i in range(1, 19)  # D1からP19まで初期化
        }

    # painditect が選ばれたときだけ、メイン画面に表示
    if choice_2 in ["BS-POP", "FUSION"]:
        st.header("使用するカラムの指定(BS-POP)")

        # 全選択・全解除ボタン
        col_buttons = st.columns(2)
        if col_buttons[0].button('全選択', key='select_all_2'):
            for key in st.session_state.checkbox_states_2:
                st.session_state.checkbox_states_2[key] = True
                st.session_state[key] = True
            st.rerun()

        if col_buttons[1].button('全解除', key='deselect_all_2'):
            for key in st.session_state.checkbox_states_2:
                st.session_state.checkbox_states_2[key] = False
                st.session_state[key] = False
            st.rerun()

        col_3 = st.columns(6)
        BSPOP_1 = col_3[0].checkbox(label='D1', value=st.session_state.checkbox_states_2["D1"], key="D1")
        BSPOP_2 = col_3[1].checkbox(label='D2', value=st.session_state.checkbox_states_2["D2"], key="D2")
        BSPOP_3 = col_3[2].checkbox(label='D3', value=st.session_state.checkbox_states_2["D3"], key="D3")
        BSPOP_4 = col_3[3].checkbox(label='D4', value=st.session_state.checkbox_states_2["D4"], key="D4")
        BSPOP_5 = col_3[4].checkbox(label='D5', value=st.session_state.checkbox_states_2["D5"], key="D5")
        BSPOP_6 = col_3[5].checkbox(label='D6', value=st.session_state.checkbox_states_2["D6"], key="D6")

        # 2行目のチェックボックス（D7〜D12）
        col_4 = st.columns(6)
        BSPOP_7 = col_4[0].checkbox(label='D7', value=st.session_state.checkbox_states_2["D7"], key="D7")
        BSPOP_8 = col_4[1].checkbox(label='D8', value=st.session_state.checkbox_states_2["D8"], key="D8")
        BSPOP_9 = col_4[2].checkbox(label='D9', value=st.session_state.checkbox_states_2["D9"], key="D9")
        BSPOP_10 = col_4[3].checkbox(label='D10', value=st.session_state.checkbox_states_2["D10"], key="D10")
        BSPOP_11 = col_4[4].checkbox(label='D11', value=st.session_state.checkbox_states_2["D11"], key="D11")
        BSPOP_12 = col_4[5].checkbox(label='D12', value=st.session_state.checkbox_states_2["D12"], key="D12")

        # 3行目のチェックボックス（D13〜D18）
        col_5 = st.columns(6)
        BSPOP_13 = col_5[0].checkbox(label='D13', value=st.session_state.checkbox_states_2["D13"], key="D13")
        BSPOP_14 = col_5[1].checkbox(label='D14', value=st.session_state.checkbox_states_2["D14"], key="D14")
        BSPOP_15 = col_5[2].checkbox(label='D15', value=st.session_state.checkbox_states_2["D15"], key="D15")
        BSPOP_16 = col_5[3].checkbox(label='D16', value=st.session_state.checkbox_states_2["D16"], key="D16")
        BSPOP_17 = col_5[4].checkbox(label='D17', value=st.session_state.checkbox_states_2["D17"], key="D17")
        BSPOP_18 = col_5[5].checkbox(label='D18', value=st.session_state.checkbox_states_2["D18"], key="D18")

        # 状態を反映
        st.session_state.checkbox_states_2["D1"] = BSPOP_1
        st.session_state.checkbox_states_2["D2"] = BSPOP_2
        st.session_state.checkbox_states_2["D3"] = BSPOP_3
        st.session_state.checkbox_states_2["D4"] = BSPOP_4
        st.session_state.checkbox_states_2["D5"] = BSPOP_5
        st.session_state.checkbox_states_2["D6"] = BSPOP_6
        st.session_state.checkbox_states_2["D7"] = BSPOP_7
        st.session_state.checkbox_states_2["D8"] = BSPOP_8
        st.session_state.checkbox_states_2["D9"] = BSPOP_9
        st.session_state.checkbox_states_2["D10"] = BSPOP_10
        st.session_state.checkbox_states_2["D11"] = BSPOP_11
        st.session_state.checkbox_states_2["D12"] = BSPOP_12
        st.session_state.checkbox_states_2["D13"] = BSPOP_13
        st.session_state.checkbox_states_2["D14"] = BSPOP_14
        st.session_state.checkbox_states_2["D15"] = BSPOP_15
        st.session_state.checkbox_states_2["D16"] = BSPOP_16
        st.session_state.checkbox_states_2["D17"] = BSPOP_17
        st.session_state.checkbox_states_2["D18"] = BSPOP_18

    st.markdown('#### 重みづけの指定')

    st.session_state.checkbox_states_1.get("P1", False)

    stocks = []
    # PainDITECT または FUSION のときだけP系を追加
    if choice_2 in ["PainDITECT", "FUSION"]:
        if st.session_state.get("P1", False):
            stocks.append('P1')
        if st.session_state.get("P2", False):
            stocks.append('P2')
        if st.session_state.get("P3", False):
            stocks.append('P3')
        if st.session_state.get("P4", False):
            stocks.append('P4')
        if st.session_state.get("P5", False):
            stocks.append('P5')
        if st.session_state.get("P6", False):
            stocks.append('P6')
        if st.session_state.get("P7", False):
            stocks.append('P7')
        if st.session_state.get("P8", False):
            stocks.append('P8')
        if st.session_state.get("P9", False):
            stocks.append('P9')
        if st.session_state.get("P10", False):
            stocks.append('P10')
        if st.session_state.get("P11", False):
            stocks.append('P11')
        if st.session_state.get("P12", False):
            stocks.append('P12')
        if st.session_state.get("P13", False):
            stocks.append('P13')

    # BS-POPまたはFUSION のときだけD系を追加
    if choice_2 in ["BS-POP", "FUSION"]:
        for i in range(1, 19):
            if st.session_state.get(f"D{i}", False):
                stocks.append(f"D{i}")


    weights = []

    # セッションステートの初期化
    if "weights" not in st.session_state:
        st.session_state.weights = {stock: 1.0 for stock in stocks}
    if "reset" not in st.session_state:
        st.session_state.reset = False

    # 重みの初期化
    if st.button("重みリセット"):
        for key in st.session_state.weights.keys():
            st.session_state.weights[key] = 1.0
        st.rerun()

    # 動的にスライドバーを生成し、weightsに格納
    st.sidebar.markdown("### 重み付け")
    for column in stocks:
        if column not in st.session_state.weights:
            st.session_state.weights[column] = 1.0
        weight = st.sidebar.slider(f"{column}の重み", min_value=-5.0, max_value=5.0, value=float(st.session_state.weights[column]), step=0.1, key=f"slider_{column}")
        weights.append(weight)
        st.session_state.weights[column] = weight

    # データフレームを作成
    edited_df = pd.DataFrame({"columns": stocks, "weights": weights})

    # データフレームを表示
    st.dataframe(edited_df)

    # st.markdown('#### データの標準化')
    # セレクトボックスのオプションを定義
    options = ['する', 'しない']

    # セレクトボックスを作成し、ユーザーの選択を取得
    choice_4 = st.sidebar.selectbox('データの標準化', options, index = None, placeholder="選択してください")

    #データの加工方法の指定
    options = ['欠損値削除', '中央値補完', '平均値補完', 'k-NN法補完']

    # セレクトボックスを作成し、ユーザーの選択を取得
    data_processing = st.sidebar.selectbox('欠損値補完の方法は？', options, index = None, placeholder="選択してください")

    if st.button("開始", help="実験の実行"):
        start_time = time.time()

        columns = edited_df["columns"].tolist()
        weights = edited_df["weights"].tolist()
        
        # データの指定
        df_nociceptive_train = df1[columns]
        df_neuronociceptive_train = df2[columns]
        df_unknown_train = df3[columns]

        # 重みを適用して特徴量を調整
        df_nociceptive_train_weighted = df_nociceptive_train.mul(weights, axis=1)
        df_neuronociceptive_train_weighted = df_neuronociceptive_train.mul(weights, axis=1)
        df_unknown_train_weighted = df_unknown_train.mul(weights, axis=1)
        
        # トレーニングデータとラベルの作成
        datas = np.vstack(
            [
                df_nociceptive_train_weighted.values,
                df_neuronociceptive_train_weighted.values,
                df_unknown_train_weighted.values,
                ]
                ).astype(np.float32)
        
        labels1 = np.full(len(df_nociceptive_train_weighted), 1, np.int32)
        labels2 = np.full(len(df_neuronociceptive_train_weighted), 2, np.int32)
        labels3 = np.full(len(df_unknown_train_weighted), 3, np.int32)
        labels = np.concatenate([labels1, labels2, labels3]).astype(np.int32)
        
        # 標準化の処理（必要に応じて）
        if choice_4 == "する":
            scaler = StandardScaler()
            datas = scaler.fit_transform(datas)

        # パラメータの候補を設定
        # === サイドバーで指定した候補値とカーネルを使ってグリッドサーチ ===
        # 既にサイドバーで C_values / gamma_values / degree_values / coef0_values / kernel / k_cv / max_iter_svc が定義済み
        best_score = -1.0
        best_params = {}
        best_model = None
        all_results = []

        # 組み合わせの生成（カーネルに応じて）
        param_grid = []
        if kernel == "linear":
            for C in C_values:
                param_grid.append({"C": C})
        elif kernel == "rbf":
            for C in C_values:
                for gamma in gamma_values:
                    param_grid.append({"C": C, "gamma": gamma})
        elif kernel == "poly":
            for C in C_values:
                for gamma in gamma_values:
                    for degree in degree_values:
                        for coef0 in coef0_values:
                            param_grid.append({"C": C, "gamma": gamma, "degree": degree, "coef0": coef0})
        elif kernel == "sigmoid":
            for C in C_values:
                for gamma in gamma_values:
                    for coef0 in coef0_values:
                        param_grid.append({"C": C, "gamma": gamma, "coef0": coef0})

        skf = StratifiedKFold(n_splits=k_cv, shuffle=True, random_state=42)

        # 各組み合わせでCV評価
        for params in param_grid:
            C = params["C"]
            gamma = params.get("gamma", "scale")
            degree = params.get("degree", 3)
            coef0 = params.get("coef0", 0.0)

            scores = []
            for train_index, val_index in skf.split(datas, labels):
                X_train, X_val = datas[train_index], datas[val_index]
                y_train, y_val = labels[train_index], labels[val_index]

                svm = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, max_iter=max_iter_svc)
                svm.fit(X_train, y_train)
                predicted_fold = svm.predict(X_val)
                score = float(np.mean(y_val == predicted_fold))
                scores.append(score)

            avg_score = float(np.mean(scores))

            # ログ用
            all_results.append({
                "kernel": kernel,
                "gamma": gamma if kernel in ["rbf", "poly", "sigmoid"] else None,
                "degree": degree if kernel == "poly" else None,
                "coef0": coef0 if kernel in ["poly", "sigmoid"] else None,
                "C": C,
                "score": avg_score,
                "weights": weights
            })

            if avg_score > best_score:
                best_score = avg_score
                best_params = {"C": C, "gamma": gamma, "degree": degree, "coef0": coef0, "kernel": kernel}
                best_model = svm

        # モデル保存
        joblib.dump(best_model, MODEL_PATH)

        elapsed = time.time() - start_time
        st.write(f"⏱ 実行時間: {elapsed:.2f} 秒")

        st.subheader("📊 スコアまとめ（降順）")
        results_df = pd.DataFrame([{
            "kernel": r["kernel"],
            "gamma": r["gamma"],
            "degree": r["degree"],
            "coef0": r["coef0"],
            "C": r["C"],
            "score": r["score"],
            "weights": r["weights"]
        } for r in all_results])
        results_df["score(%)"] = (results_df["score"] * 100).map(lambda x: f"{x:.2f}%")
        st.dataframe(results_df.sort_values(by="score", ascending=False))

        # モデル読み込み
        svm = joblib.load(MODEL_PATH)
        predicted = svm.predict(X_val)

        st.write(f"✅ 最終スコア: {best_score * 100:.2f}%")
            
        # 感度と特異度の計算
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
            
        # # 感度と特異度の表示
        # st.write("感度と特異度")
        # st.write("（疼痛1:侵害受容性疼痛,疼痛2:神経障害性疼痛,疼痛3:不明）")
        # for i in range(3):
        #     st.write(f"疼痛 {i+1}: 感度 = {sensitivity[i]:.4f}, 特異度 = {specificity[i]:.4f}")

        # 現在の日時を取得
        dt_now = datetime.datetime.now()

        # アップロードしたCSVファイルのパス
        LOG_FILE_PATH = 'log/LOG_FILE.csv'

        # 新しいデータを1行にまとめる
        new_row = {
            'date': dt_now.strftime('%Y%m%d-%H%M%S'),
            'data_processing': data_processing,
            'use_columns': ', '.join(map(str, columns)),
            'weights': ', '.join(map(str, weights)),
            'score': str(best_score*100),
            'sensitivity': ', '.join(f"{x:.4f}" for x in sensitivity_list),
            'specificity': ', '.join(f"{x:.4f}" for x in specificity_list)
        }

        # CSVファイルに追記（既存のヘッダーを維持）
        with open(LOG_FILE_PATH, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=new_row.keys())

            # データを一行で追加
            writer.writerow(new_row)