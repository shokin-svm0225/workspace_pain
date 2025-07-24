import streamlit as st
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from concurrent.futures import ThreadPoolExecutor
import time
import random

MODEL_PATH = "svm_model.pkl"

def run_secondshift_experiment():
    st.title('実験')

    with st.container(border=True):
        col1, col2 = st.columns(2)
    # 各カラムに画像を表示
        with col1:
            # with st.container(border=True):
            st.subheader('山登り法', divider='rainbow')
            st.markdown("""
            - グローバルベスト \n
            各特徴量ごとに「+ε/-ε/±0の三方向」（現在までのベストスコアを考慮）で正答率を出し、3×n(特徴量)通りの中で一番良い方向に更新していく
            """)
        with col2:
            st.code("""
            重み = [1, 1, 1, 1, 1]   ← 初期状態  
            ↓  
            各特徴量について  
                重み + [-ε, 0, +ε](delta) の3通りを試す  
                ・delta = 0 のときは評価せず、今のベストスコアを使う  
                → スコアが最も良い重みを記録  
            ↓  
            全特徴量を一巡したら一番良かった重みに更新  
            ↓  
            これを max_iter 回繰り返す
            """, language="text")

    # セレクトボックスのオプションを定義
    options = ['欠損値データ削除', '中央値補完', '平均値補完', 'k-NN法補完']

    # セレクトボックスを作成し、ユーザーの選択を取得
    choice_1 = st.sidebar.selectbox('欠損値の対応', options, index = None, placeholder="選択してください")

    # セレクトボックスのオプションを定義
    options = ['PainDITECT', 'BS-POP', 'FUSION']

    # セレクトボックスを作成し、ユーザーの選択を取得
    choice_2 = st.sidebar.selectbox('使用する質問表', options, index = None, placeholder="選択してください")

    # セレクトボックスのオプションを定義
    options = ['有', '無']

    # セレクトボックスを作成し、ユーザーの選択を取得
    choice_3 = st.sidebar.selectbox('特徴量拡大の有無', options, index = None, placeholder="選択してください")

    if choice_1 == '欠損値データ削除' and choice_2 == 'PainDITECT' and choice_3 == '無':
        df1 = pd.read_csv('data/null/peindetect/questionnaire_paindetect_missing_侵害.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/null/peindetect/questionnaire_paindetect_missing_神経.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/null/peindetect/questionnaire_paindetect_missing_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '欠損値データ削除' and choice_2 == 'BS-POP' and choice_3 == '無':
        df1 = pd.read_csv('data/null/BSPOP/questionnaire_bspop_missing_侵害.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/null/BSPOP/questionnaire_bspop_missing_神経.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/null/BSPOP/questionnaire_bspop_missing_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '欠損値データ削除' and choice_2 == 'FUSION' and choice_3 == '無':
        df1 = pd.read_csv('data/null/fusion/questionnaire_fusion_missing_侵害.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/null/fusion/questionnaire_fusion_missing_神経.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/null/fusion/questionnaire_fusion_missing_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'PainDITECT' and choice_3 == '無':
        df1 = pd.read_csv('data/欠損値補完/PAINDITECT/det_median_侵害受容性疼痛_paindetect.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/PAINDITECT/det_median_神経障害性疼痛_paindetect.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/PAINDITECT/det_median_不明_paindetect.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'BS-POP' and choice_3 == '無':
        df1 = pd.read_csv('data/欠損値補完/BSPOP/det_median_侵害受容性疼痛_bspop.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/BSPOP/det_median_神経障害性疼痛_bspop.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/BSPOP/det_median_不明_bspop.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'FUSION' and choice_3 == '無':
        df1 = pd.read_csv('data/欠損値補完/FUSION/det_median_侵害受容性疼痛.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/FUSION/det_median_神経障害性疼痛.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/FUSION/det_median_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'PainDITECT' and choice_3 == '無':
        df1 = pd.read_csv('data/欠損値補完/PAINDITECT/det_mean_侵害受容性疼痛_paindetect.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/PAINDITECT/det_mean_神経障害性疼痛_paindetect.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/PAINDITECT/det_mean_不明_paindetect.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'BS-POP' and choice_3 == '無':
        df1 = pd.read_csv('data/欠損値補完/BSPOP/det_mean_侵害受容性疼痛_bspop.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/BSPOP/det_mean_神経障害性疼痛_bspop.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/BSPOP/det_mean_不明_bspop.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'FUSION' and choice_3 == '無':
        df1 = pd.read_csv('data/欠損値補完/FUSION/det_mean_侵害受容性疼痛.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/FUSION/det_mean_神経障害性疼痛.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/FUSION/det_mean_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'PainDITECT' and choice_3 == '無':
        df1 = pd.read_csv('data/欠損値補完/PAINDITECT/det_KNN_侵害受容性疼痛_paindetect.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/PAINDITECT/det_KNN_神経障害性疼痛_paindetect.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/PAINDITECT/det_KNN_不明_paindetect.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'BS-POP' and choice_3 == '無':
        df1 = pd.read_csv('data/欠損値補完/BSPOP/det_KNN_侵害受容性疼痛_bspop.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/BSPOP/det_KNN_神経障害性疼痛_bspop.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/BSPOP/det_KNN_不明_bspop.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'FUSION' and choice_3 == '無':
        df1 = pd.read_csv('data/欠損値補完/FUSION/det_KNN_侵害受容性疼痛.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data/欠損値補完/FUSION/det_KNN_神経障害性疼痛.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data/欠損値補完/FUSION/det_KNN_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '欠損値データ削除' and choice_2 == 'PainDITECT' and choice_3 == '有':
        df1 = pd.read_csv('data2/特徴量拡大/painDETECT/NULL/侵害受容性疼痛_filtered_data_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data2/特徴量拡大/painDETECT/NULL/神経障害性疼痛_filtered_data_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data2/特徴量拡大/painDETECT/NULL/不明_filtered_data_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '欠損値データ削除' and choice_2 == 'BS-POP' and choice_3 == '有':
        df1 = pd.read_csv('data2/特徴量拡大/BS-POP/NULL/questionnaire_bspop_missing_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data2/特徴量拡大/BS-POP/NULL/questionnaire_bspop_missing_精神障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data2/特徴量拡大/BS-POP/NULL/questionnaire_bspop_missing_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '欠損値データ削除' and choice_2 == 'FUSION' and choice_3 == '有':
        df1 = pd.read_csv('data2/特徴量拡大/FUSION/NULL/questionnaire_fusion_missing_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data2/特徴量拡大/FUSION/NULL/questionnaire_fusion_missing_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data2/特徴量拡大/FUSION/NULL/questionnaire_fusion_missing_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'PainDITECT' and choice_3 == '有':
        df1 = pd.read_csv('data2/特徴量拡大/painDETECT/median/det_painditect_median_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data2/特徴量拡大/painDETECT/median/det_painditect_median_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data2/特徴量拡大/painDETECT/median/det_painditect_median_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'BS-POP' and choice_3 == '有':
        df1 = pd.read_csv('data2/特徴量拡大/BS-POP/median/det_bspop_median_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data2/特徴量拡大/BS-POP/median/det_bspop_median_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data2/特徴量拡大/BS-POP/median/det_bspop_median_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'FUSION' and choice_3 == '有':
        df1 = pd.read_csv('data2/特徴量拡大/FUSION/median/det_median_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data2/特徴量拡大/FUSION/median/det_median_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data2/特徴量拡大/FUSION/median/det_median_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'PainDITECT' and choice_3 == '有':
        df1 = pd.read_csv('data2/特徴量拡大/painDETECT/mean/det_painditect_mean_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data2/特徴量拡大/painDETECT/mean/det_painditect_mean_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data2/特徴量拡大/painDETECT/mean/det_painditect_mean_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'BS-POP' and choice_3 == '有':
        df1 = pd.read_csv('data2/特徴量拡大/BS-POP/mean/det_bspop_mean_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data2/特徴量拡大/BS-POP/mean/det_bspop_mean_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data2/特徴量拡大/BS-POP/mean/det_bspop_mean_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'FUSION' and choice_3 == '有':
        df1 = pd.read_csv('data2/特徴量拡大/FUSION/mean/det_mean_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data2/特徴量拡大/FUSION/mean/det_mean_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data2/特徴量拡大/FUSION/mean/det_mean_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'PainDITECT' and choice_3 == '有':
        df1 = pd.read_csv('data2/特徴量拡大/painDETECT/knn/det_painditect_KNN_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data2/特徴量拡大/painDETECT/knn/det_painditect_KNN_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data2/特徴量拡大/painDETECT/knn/det_painditect_KNN_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'BS-POP' and choice_3 == '有':
        df1 = pd.read_csv('data2/特徴量拡大/BS-POP/knn/det_bspop_KNN_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data2/特徴量拡大/BS-POP/knn/det_bspop_KNN_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data2/特徴量拡大/BS-POP/knn/det_bspop_KNN_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'FUSION' and choice_3 == '有':
        df1 = pd.read_csv('data2/特徴量拡大/FUSION/knn/det_KNN_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('data2/特徴量拡大/FUSION/knn/det_KNN_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('data2/特徴量拡大/FUSION/knn/det_KNN_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

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

        if col_buttons[1].button('全解除', key='deselect_all_1'):
            for key in st.session_state.checkbox_states_1:
                st.session_state.checkbox_states_1[key] = False

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

        if col_buttons[1].button('全解除', key='deselect_all_2'):
            for key in st.session_state.checkbox_states_2:
                st.session_state.checkbox_states_2[key] = False

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

    if choice_3 == "有":
        st.header("使用するカラムの指定(特徴量拡大)")

        if 'checkbox_states_3' not in st.session_state:
            st.session_state.checkbox_states_3 = {
                f"S{i}": False for i in range(1, 5)  # P1からP7まで初期化
            }

        # 全選択・全解除ボタン
        col_buttons = st.columns(2)
        if col_buttons[0].button('全選択', key='select_all_3'):
            for key in st.session_state.checkbox_states_3:
                st.session_state.checkbox_states_3[key] = True

        if col_buttons[1].button('全解除', key='deselect_all_3'):
            for key in st.session_state.checkbox_states_3:
                st.session_state.checkbox_states_3[key] = False

        # チェックボックスの表示（元のスタイルを維持）
        col_6 = st.columns(4)
        expand_1 = col_6[0].checkbox(label='S1', value=st.session_state.checkbox_states_3["S1"], key="S1")
        expand_2 = col_6[1].checkbox(label='S2', value=st.session_state.checkbox_states_3["S2"], key="S2")
        expand_3 = col_6[2].checkbox(label='S3', value=st.session_state.checkbox_states_3["S3"], key="S3")
        expand_4 = col_6[3].checkbox(label='S4', value=st.session_state.checkbox_states_3["S4"], key="S4")

        # 状態を反映
        st.session_state.checkbox_states_3["S1"] = expand_1
        st.session_state.checkbox_states_3["S2"] = expand_2
        st.session_state.checkbox_states_3["S3"] = expand_3
        st.session_state.checkbox_states_3["S4"] = expand_4

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
    if st.button("重みをリセット", key="weights_reset"):
        for stock in stocks:
            st.session_state.weights[stock] = 1.0  # 全ての重みを初期化
        st.session_state.reset = True

    # 動的にスライドバーを生成し、weightsに格納
    for column in stocks:
        if column not in st.session_state.weights:
            st.session_state.weights[column] = 1.0
        # セッションステートからスライダーの初期値を取得
        default_weight = st.session_state.weights[column]
        st.sidebar.markdown("### 重み付け")
        weight = st.sidebar.slider(f"{column}の重み", min_value=-5.0, max_value=5.0, value=default_weight, step=0.1, key=f"slider_{column}")
        weights.append(weight)
        # スライダーの値をセッションステートに保存
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

        initial_weights = np.random.randint(-5, 5, datas.shape[1]).astype(float)

        # === 共通関数群 ===
        def apply_weights(datas, weights_change):
            return datas * weights_change

        def evaluate(weights_change, datas, labels, C, gamma=0.1, coef0=0.0, k=5, return_best_split=False):
            X_weighted = apply_weights(datas, weights_change)
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            scores = []

            best_fold_score = 0
            best_X_val, best_y_val, best_pred = None, None, None

            for train_index, val_index in skf.split(X_weighted, labels):
                X_train, X_val = X_weighted[train_index], X_weighted[val_index]
                y_train, y_val = labels[train_index], labels[val_index]

                model = SVC(C=C, kernel='sigmoid', gamma=gamma, coef0=coef0, max_iter=1500)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                acc = np.mean(y_pred == y_val)
                scores.append(acc)

                if return_best_split and acc > best_fold_score:
                    best_fold_score = acc
                    best_X_val = X_val
                    best_y_val = y_val
                    best_pred = y_pred

            if return_best_split:
                return np.mean(scores), best_X_val, best_y_val, best_pred
            else:
                return np.mean(scores)

        def hill_climbing(datas, labels, C, gamma, coef0, initial_weights, max_iter_1=30, step_size=0.01):
            n_features = datas.shape[1]
            # weights_change = np.ones(n_features).astype(float)
            weights_change = initial_weights.copy()  # 外から渡された固定の初期重み

            best_score, best_X_val, best_y_val, best_pred = evaluate(weights_change, datas, labels, C, gamma, coef0, return_best_split=True)
            best_weights = weights_change.copy()
            score_history = [best_score]

            for i in range(max_iter_1):
                step_best_score = -np.inf 
                candidates = [] 

                for idx in range(n_features):
                    for delta in [-step_size, step_size]:
                        trial_weights = weights_change.copy()
                        trial_weights = trial_weights.astype(float)
                        trial_weights[idx] += delta

                        score, X_val_tmp, y_val_tmp, pred_tmp = evaluate(
                            trial_weights, datas, labels, C, gamma, coef0, return_best_split=True
                        )

                    if score > step_best_score:
                        step_best_score = score
                        candidates = [(trial_weights.copy(), X_val_tmp, y_val_tmp, pred_tmp)]  # 🔄 新しく記録
                    elif score == step_best_score:
                        candidates.append((trial_weights.copy(), X_val_tmp, y_val_tmp, pred_tmp)) 

                # ✅ スコアが同じ候補からランダムに1つを選ぶ
                selected_weights, selected_X_val, selected_y_val, selected_pred = random.choice(candidates)
                weights_change = selected_weights
                best_weights = weights_change.copy()
                best_score = step_best_score
                best_X_val, best_y_val, best_pred = selected_X_val, selected_y_val, selected_pred
                score_history.append(best_score)

            return best_weights, best_score, best_X_val, best_y_val, best_pred, score_history

        def run_hill_climbing(step_size, gamma, coef0, C, datas, labels):
            weights_change, score, X_val_tmp, y_val_tmp, pred_tmp, score_history = hill_climbing(
                datas, labels, C, gamma, coef0, initial_weights, max_iter_1=30, step_size=step_size
            )
            return {
                "step_size": step_size,
                "gamma": gamma,
                "coef0": coef0,
                "C": C,
                "score": score,
                "weights": [float(f"{w:.2f}") for w in weights_change],
                "score_history": score_history,
                "X_val": X_val_tmp,
                "y_val": y_val_tmp,
                "pred": pred_tmp,
            }

        # === パラメータ設定 & 実行 ===
        st.title("🧠 Hill Climbing × 並列探索（SVM最適化）")

        step_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
        C_values = [0.01, 0.1, 1]
        gamma_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        coef0_values = [-5, -2, 0, 2, 5]

        param_grid = [
            (step_size, gamma, coef0, C)
            for step_size in step_sizes
            for gamma in gamma_values
            for coef0 in coef0_values
            for C in C_values
        ]

        all_results = []
        best_score = 0
        best_result = None

        st.write("🔁 並列実行中...")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(run_hill_climbing, step_size, gamma, coef0, C, datas, labels)
                for (step_size, gamma, coef0, C) in param_grid
            ]

        results = [f.result() for f in futures]

        for result in results:
            all_results.append(result)

            if result["score"] > best_score:
                best_score = result["score"]
                best_result = result

        # 結果表示
        results_df = pd.DataFrame([{
            "step_size": r["step_size"],
            "gamma": r["gamma"],
            "coef0": r["coef0"],
            "C": r["C"],
            "score": r["score"],
            "weights": r["weights"]
        } for r in all_results])


        elapsed = time.time() - start_time
        st.write(f"⏱ 実行時間: {elapsed:.2f} 秒")

        st.subheader("📊 スコアまとめ")
        st.dataframe(results_df.sort_values(by="score", ascending=False))

        st.subheader("📊 一番良かったパラメータのスコア推移")

        best_history = best_result["score_history"]

        fig, ax = plt.subplots()
        ax.plot(range(len(best_history)), best_history)
        ax.set_title("Best Score Progression by Hill Climbing")
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        st.pyplot(fig)

        # モデル保存
        if best_result:
            X_weighted_final = apply_weights(datas, np.array(best_result["weights"]))
            final_model = SVC(
                C=best_result["C"],
                kernel='sigmoid',
                gamma=best_result["gamma"],
                coef0=best_result["coef0"],
                max_iter=1500
            )
            final_model.fit(X_weighted_final, labels)
            joblib.dump(final_model, "final_model.joblib")
            st.success("✅ 最終モデルを保存しました！")

            # データフレームを作成
            # best_weights_df = pd.DataFrame(best_weights.astype(float),{"columns": stocks, "weights": best_weights})
            best_weights_df = pd.DataFrame(np.array(best_result["weights"]).astype(float), index=stocks, columns=["Weight"])

            # ✅ ここにスコア一覧表を表示
            st.subheader("📊 step_size × C ごとのスコアまとめ")
            results_df = pd.DataFrame([
                {"step_size": r["step_size"], "C": r["C"], "score": r["score"]}
                for r in all_results
            ])
            results_df["score"] = (results_df["score"] * 100).map(lambda x: f"{x:.2f}%")
            st.dataframe(results_df)

            # 結果表示
            st.write("✅ 最適なC:", best_result["C"])
            st.write("✅ 最適な重み:")
            st.dataframe(best_weights_df)
            st.write("✅ 最終スコア:", best_score)

            best_y_val = best_result["y_val"]
            best_pred = best_result["pred"]

            # 感度と特異度の計算
            conf_matrix = confusion_matrix(best_y_val, best_pred, labels=[1, 2, 3])

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