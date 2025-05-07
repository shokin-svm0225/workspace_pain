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

TEST_DATA_RATIO = 0.3
SAVE_TRAINED_DATA_PATH = "svm_data.xml"

def show():
    st.title('疼痛診断システムの開発')
    st.markdown('#### 欠損値の対応')
    # セレクトボックスのオプションを定義
    options = ['欠損値データ削除', '中央値補完', '平均値補完', 'k-NN法補完']

    # セレクトボックスを作成し、ユーザーの選択を取得
    choice_1 = st.selectbox('欠損値の対応は？', options, index = None, placeholder="選択してください")

    # ユーザーの選択に応じたメッセージを表示
    st.write(f'あなたが選んだのは  {choice_1}  です。')

    st.markdown('#### 使用する質問表')
    # セレクトボックスのオプションを定義
    options = ['PainDITECT', 'BS-POP', 'FUSION']

    # セレクトボックスを作成し、ユーザーの選択を取得
    choice_2 = st.selectbox('使用する質問表は？', options, index = None, placeholder="選択してください")

    # ユーザーの選択に応じたメッセージを表示
    st.write(f'あなたが選んだのは  {choice_2}  です。')

    st.markdown('#### 特徴量拡大の有無')
    # セレクトボックスのオプションを定義
    options = ['有', '無']

    # セレクトボックスを作成し、ユーザーの選択を取得
    choice_3 = st.selectbox('特徴量拡大を行いますか？', options, index = None, placeholder="選択してください")

    # ユーザーの選択に応じたメッセージを表示
    st.write(f'あなたが選んだのは  {choice_3}  です。')

    if choice_1 == '欠損値データ削除' and choice_2 == 'PainDITECT' and choice_3 == '無':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/null/peindetect/questionnaire_paindetect_missing_侵害.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/null/peindetect/questionnaire_paindetect_missing_神経.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/null/peindetect/questionnaire_paindetect_missing_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '欠損値データ削除' and choice_2 == 'BS-POP' and choice_3 == '無':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/null/BSPOP/questionnaire_bspop_missing_侵害.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/null/BSPOP/questionnaire_bspop_missing_神経.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/null/BSPOP/questionnaire_bspop_missing_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '欠損値データ削除' and choice_2 == 'FUSION' and choice_3 == '無':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/null/fusion/questionnaire_fusion_missing_侵害.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/null/fusion/questionnaire_fusion_missing_神経.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/null/fusion/questionnaire_fusion_missing_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'PainDITECT' and choice_3 == '無':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/PAINDITECT/det_median_侵害受容性疼痛_paindetect.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/PAINDITECT/det_median_神経障害性疼痛_paindetect.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/PAINDITECT/det_median_不明_paindetect.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'BS-POP' and choice_3 == '無':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/BSPOP/det_median_侵害受容性疼痛_bspop.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/BSPOP/det_median_神経障害性疼痛_bspop.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/BSPOP/det_median_不明_bspop.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'FUSION' and choice_3 == '無':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/FUSION/det_median_侵害受容性疼痛.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/FUSION/det_median_神経障害性疼痛.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/FUSION/det_median_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'PainDITECT' and choice_3 == '無':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/PAINDITECT/det_mean_侵害受容性疼痛_paindetect.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/PAINDITECT/det_mean_神経障害性疼痛_paindetect.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/PAINDITECT/det_mean_不明_paindetect.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'BS-POP' and choice_3 == '無':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/BSPOP/det_mean_侵害受容性疼痛_bspop.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/BSPOP/det_mean_神経障害性疼痛_bspop.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/BSPOP/det_mean_不明_bspop.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'FUSION' and choice_3 == '無':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/FUSION/det_mean_侵害受容性疼痛.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/FUSION/det_mean_神経障害性疼痛.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/FUSION/det_mean_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'PainDITECT' and choice_3 == '無':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/PAINDITECT/det_KNN_侵害受容性疼痛_paindetect.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/PAINDITECT/det_KNN_神経障害性疼痛_paindetect.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/PAINDITECT/det_KNN_不明_paindetect.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'BS-POP' and choice_3 == '無':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/BSPOP/det_KNN_侵害受容性疼痛_bspop.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/BSPOP/det_KNN_神経障害性疼痛_bspop.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/BSPOP/det_KNN_不明_bspop.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'FUSION' and choice_3 == '無':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/FUSION/det_KNN_侵害受容性疼痛.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/FUSION/det_KNN_神経障害性疼痛.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/欠損値補完/FUSION/det_KNN_不明.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '欠損値データ削除' and choice_2 == 'PainDITECT' and choice_3 == '有':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/painDETECT/NULL/侵害受容性疼痛_filtered_data_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/painDETECT/NULL/神経障害性疼痛_filtered_data_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/painDETECT/NULL/不明_filtered_data_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '欠損値データ削除' and choice_2 == 'BS-POP' and choice_3 == '有':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/BS-POP/NULL/questionnaire_bspop_missing_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/BS-POP/NULL/questionnaire_bspop_missing_精神障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/BS-POP/NULL/questionnaire_bspop_missing_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '欠損値データ削除' and choice_2 == 'FUSION' and choice_3 == '有':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/FUSION/NULL/questionnaire_fusion_missing_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/FUSION/NULL/questionnaire_fusion_missing_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/FUSION/NULL/questionnaire_fusion_missing_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'PainDITECT' and choice_3 == '有':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/painDETECT/median/det_painditect_median_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/painDETECT/median/det_painditect_median_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/painDETECT/median/det_painditect_median_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'BS-POP' and choice_3 == '有':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/BS-POP/median/det_bspop_median_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/BS-POP/median/det_bspop_median_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/BS-POP/median/det_bspop_median_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '中央値補完' and choice_2 == 'FUSION' and choice_3 == '有':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/FUSION/median/det_median_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/FUSION/median/det_median_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/FUSION/median/det_median_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'PainDITECT' and choice_3 == '有':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/painDETECT/mean/det_painditect_mean_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/painDETECT/mean/det_painditect_mean_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/painDETECT/mean/det_painditect_mean_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'BS-POP' and choice_3 == '有':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/BS-POP/mean/det_bspop_mean_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/BS-POP/mean/det_bspop_mean_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/BS-POP/mean/det_bspop_mean_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == '平均値補完' and choice_2 == 'FUSION' and choice_3 == '有':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/FUSION/mean/det_mean_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/FUSION/mean/det_mean_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/FUSION/mean/det_mean_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'PainDITECT' and choice_3 == '有':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/painDETECT/knn/det_painditect_KNN_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/painDETECT/knn/det_painditect_KNN_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/painDETECT/knn/det_painditect_KNN_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'BS-POP' and choice_3 == '有':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/BS-POP/knn/det_bspop_KNN_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/BS-POP/knn/det_bspop_KNN_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/BS-POP/knn/det_bspop_KNN_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    elif choice_1 == 'k-NN法補完' and choice_2 == 'FUSION' and choice_3 == '有':
        df1 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/FUSION/knn/det_KNN_侵害受容性疼痛_newroc.csv', encoding = 'utf-8')
        df2 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/FUSION/knn/det_KNN_神経障害性疼痛_newroc.csv', encoding = 'utf-8')
        df3 = pd.read_csv('/Users/iwasho_0225/Desktop/workspace/pain_experiment/特徴量拡大/FUSION/knn/det_KNN_不明_newroc.csv', encoding = 'utf-8')
        st.markdown('#### 侵害受容性疼痛')
        st.dataframe(df1)
        st.markdown('#### 神経障害性疼痛')
        st.dataframe(df2)
        st.markdown('#### 原因不明')
        st.dataframe(df3)

    #svmのプログラムを組み込む
    st.markdown('#### 使用するカラムの指定(painDETECT)')

    # 初期化
    if 'checkbox_states_1' not in st.session_state:
        st.session_state.checkbox_states_1 = {
            f"P{i}": False for i in range(1, 14)  # P1からP7まで初期化
        }

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

    st.markdown('#### 使用するカラムの指定(BSPOP)')

    # 初期化
    if 'checkbox_states_2' not in st.session_state:
        st.session_state.checkbox_states_2 = {
            f"D{i}": False for i in range(1, 19)  # D1からP19まで初期化
        }

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

    st.markdown('#### 使用するカラムの指定(特徴量拡大)')

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

    stocks = []
    if painDITECT_1:
        stocks.append('P1')
    if painDITECT_2:
        stocks.append('P2')
    if painDITECT_3:
        stocks.append('P3')
    if painDITECT_4:
        stocks.append('P4')
    if painDITECT_5:
        stocks.append('P5')
    if painDITECT_6:
        stocks.append('P6')
    if painDITECT_7:
        stocks.append('P7')
    if painDITECT_8:
        stocks.append('P8')
    if painDITECT_9:
        stocks.append('P9')
    if painDITECT_10:
        stocks.append('P10')
    if painDITECT_11:
        stocks.append('P11')
    if painDITECT_12:
        stocks.append('P12')
    if painDITECT_13:
        stocks.append('P13')
    if BSPOP_1:
        stocks.append('D1')
    if BSPOP_2:
        stocks.append('D2')
    if BSPOP_3:
        stocks.append('D3')
    if BSPOP_4:
        stocks.append('D4')
    if BSPOP_5:
        stocks.append('D5')
    if BSPOP_6:
        stocks.append('D6')
    if BSPOP_7:
        stocks.append('D7')
    if BSPOP_8:
        stocks.append('D8')
    if BSPOP_9:
        stocks.append('D9')
    if BSPOP_10:
        stocks.append('D10')
    if BSPOP_11:
        stocks.append('D11')
    if BSPOP_12:
        stocks.append('D12')
    if BSPOP_13:
        stocks.append('D13')
    if BSPOP_14:
        stocks.append('D14')
    if BSPOP_15:
        stocks.append('D15')
    if BSPOP_16:
        stocks.append('D16')
    if BSPOP_17:
        stocks.append('D17')
    if BSPOP_18:
        stocks.append('D18')
    if expand_1:
        stocks.append('S1')
    if expand_2:
        stocks.append('S2')
    if expand_3:
        stocks.append('S3')
    if expand_4:
        stocks.append('S4')

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
        weight = st.slider(f"{column}の重み", min_value=-5.0, max_value=5.0, value=default_weight, step=0.1, key=f"slider_{column}")
        weights.append(weight)
        # スライダーの値をセッションステートに保存
        st.session_state.weights[column] = weight

    # データフレームを作成
    edited_df = pd.DataFrame({"columns": stocks, "weights": weights})

    # データフレームを表示
    st.markdown("#### 重みづけデータフレーム")
    st.dataframe(edited_df)

    st.markdown('#### データの標準化')
    # セレクトボックスのオプションを定義
    options = ['する', 'しない']

    # セレクトボックスを作成し、ユーザーの選択を取得
    choice_4 = st.selectbox('データの標準化を行いますか？', options, index = None, placeholder="選択してください")

    # ユーザーの選択に応じたメッセージを表示
    st.write(f'あなたが選んだのは  {choice_4}  です。')

    #データの加工方法の指定
    options = ['欠損値削除', '中央値補完', '平均値補完', 'k-NN法補完']

    # セレクトボックスを作成し、ユーザーの選択を取得
    data_processing = st.selectbox('欠損値補完の方法は？', options, index = None, placeholder="選択してください")

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

        # パラメータの候補を設定
        # gamma_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000,10000] 
        C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] # 0.0001から10000までの範囲、ステップ幅1
        k = 5
        best_score = 0
        best_params = None

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)

        for C in C_values:
            scores = []

            for train_index, val_index in skf.split(datas, labels):

                X_train = datas[train_index]
                y_train = labels[train_index]
                X_val = datas[val_index]
                y_val = labels[val_index]

                svm = cv2.ml.SVM_create()
                svm.setType(cv2.ml.SVM_C_SVC)
                svm.setKernel(cv2.ml.SVM_LINEAR)
                svm.setC(C)
                svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1500, 1.0e-06))

                # トレーニング
                svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

                # バリデーションデータで評価
                _, predicted = svm.predict(X_val)
                score = np.mean(y_val == predicted.flatten())
                scores.append(score)

            avg_score = np.mean(scores)

            st.write(f"C: {C}, Score: {avg_score:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_params = {"C": C}

            # モデルを保存
            svm.save(SAVE_TRAINED_DATA_PATH)

        st.write("最適なパラメータ:", best_params)
        st.write("最高スコア:", best_score)

        svm = cv2.ml.SVM_load(SAVE_TRAINED_DATA_PATH)
        
        # confusion_matrix = np.zeros((3, 3), dtype=int)
        
        # for i in range(len(test_labels)):
        #     index1 = test_labels[i] - 1
        #     index2 = predicted[i][0] - 1
        #     confusion_matrix[int(index1)][int(index2)] += 1
            
        # st.write("confusion matrix")
        # st.table(confusion_matrix)

        # score = np.sum(test_labels == predicted.flatten()) / len(test_labels)
            
        # st.write("正答率:", score*100, "%")
            
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
        LOG_FILE_PATH = '/Users/iwasho_0225/Desktop/workspace/pain_experiment/log/LOG_FILE.csv'

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