import streamlit as st
import itertools
import plotly.express as px
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
from streamlit_option_menu import option_menu

st.title('データセットの表示')

# ラジオボタンを表示
home_type = st.sidebar.radio("選んでください", ["生データ", "欠損値削除", "中央値補完", "平均値補完", "k-NN法補完"])

if home_type == "生データ":
    st.subheader('生データ', divider='rainbow')
    df1 = pd.read_csv("data2/data_main/MedicalData_columns_change.csv", encoding = 'utf-8')
    st.dataframe(df1)

if home_type == "欠損値削除":
    st.subheader('欠損値削除', divider='rainbow')
    df2 = pd.read_csv("data/null/fusion/questionnaire_fusion_missing.csv", encoding = 'utf-8')
    st.dataframe(df2)

if home_type == "中央値補完":
    st.subheader('中央値補完', divider='rainbow')
    df3 = pd.read_csv("data/欠損値補完/FUSION/det_median_侵害受容性疼痛.csv", encoding = 'utf-8')
    df4 = pd.read_csv("data/欠損値補完/FUSION/det_median_神経障害性疼痛.csv", encoding = 'utf-8')
    df5 = pd.read_csv("data/欠損値補完/FUSION/det_median_不明.csv", encoding = 'utf-8')
    st.dataframe(df3)
    st.dataframe(df4)
    st.dataframe(df5)

if home_type == "平均値補完":
    st.subheader('平均値補完', divider='rainbow')
    df6 = pd.read_csv("data/欠損値補完/FUSION/det_mean_侵害受容性疼痛.csv", encoding = 'utf-8')
    df7 = pd.read_csv("data/欠損値補完/FUSION/det_mean_神経障害性疼痛.csv", encoding = 'utf-8')
    df8 = pd.read_csv("data/欠損値補完/FUSION/det_mean_不明.csv", encoding = 'utf-8')
    st.dataframe(df6)
    st.dataframe(df7)
    st.dataframe(df8)

if home_type == "k-NN法補完":
    st.subheader('k-NN法補完', divider='rainbow')
    df9 = pd.read_csv("data/欠損値補完/FUSION/det_KNN_侵害受容性疼痛.csv", encoding = 'utf-8')
    df10 = pd.read_csv("data/欠損値補完/FUSION/det_KNN_神経障害性疼痛.csv", encoding = 'utf-8')
    df11 = pd.read_csv("data/欠損値補完/FUSION/det_KNN_不明.csv", encoding = 'utf-8')
    st.dataframe(df9)
    st.dataframe(df10)
    st.dataframe(df11)