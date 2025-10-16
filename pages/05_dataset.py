import streamlit as st
import itertools
import plotly.express as px
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
from streamlit_option_menu import option_menu

st.title('データセット')

# ラジオボタンを表示
home_type = st.sidebar.radio("選んでください", ["生データ", "欠損値削除", "中央値補完", "平均値補完", "k-NN法補完"])

if home_type == "生データ":
    st.subheader('生データ(合計あり)', divider='rainbow')
    df1 = pd.read_csv("data2/data_main/MedicalData.csv", encoding = 'utf-8')
    st.dataframe(df1)
    csv = df1.to_csv(index=False).encode('utf-8')
    st.download_button(
    label="ダウンロード",
    data=csv,
    file_name='MedicalData.csv',
    mime='text/csv',
    )
    st.subheader('生データ(合計なし)', divider='rainbow')
    df2 = pd.read_csv("data2/data_main/MedicalData_columns_change.csv", encoding = 'utf-8')
    st.dataframe(df2)
    csv2 = df2.to_csv(index=False).encode('utf-8')
    st.download_button(
    label="ダウンロード",
    data=csv2,
    file_name='MedicalData_columns_change.csv',
    mime='text/csv',
    )

if home_type == "欠損値削除":
    st.subheader('欠損値削除', divider='rainbow')
    df2 = pd.read_csv("data/null/fusion/questionnaire_fusion_missing.csv", encoding = 'utf-8')
    st.dataframe(df2)
    csv = df2.to_csv(index=False).encode('utf-8')
    st.download_button(
    label="ダウンロード",
    data=csv,
    file_name='questionnaire_fusion_missing.csv',
    mime='text/csv',
    )

if home_type == "中央値補完":
    st.subheader('中央値補完', divider='rainbow')
    df3 = pd.read_csv("data/欠損値補完/FUSION/det_median_侵害受容性疼痛.csv", encoding = 'utf-8')
    df4 = pd.read_csv("data/欠損値補完/FUSION/det_median_神経障害性疼痛.csv", encoding = 'utf-8')
    df5 = pd.read_csv("data/欠損値補完/FUSION/det_median_不明.csv", encoding = 'utf-8')
    st.markdown("①")
    st.dataframe(df3)
    csv_1 = df3.to_csv(index=False).encode('utf-8')
    st.markdown("②")
    st.dataframe(df4)
    csv_2 = df4.to_csv(index=False).encode('utf-8')
    st.markdown("③")
    st.dataframe(df5)
    csv_3 = df5.to_csv(index=False).encode('utf-8')
    st.download_button(
    label="①をダウンロード",
    data=csv_1,
    file_name='det_median_侵害受容性疼痛.csv',
    mime='text/csv',
    )
    st.download_button(
    label="②をダウンロード",
    data=csv_2,
    file_name='det_median_神経障害性疼痛.csv',
    mime='text/csv',
    )
    st.download_button(
    label="③をダウンロード",
    data=csv_3,
    file_name='det_median_不明.csv',
    mime='text/csv',
    )

if home_type == "平均値補完":
    st.subheader('平均値補完', divider='rainbow')
    df6 = pd.read_csv("data/欠損値補完/FUSION/det_mean_侵害受容性疼痛.csv", encoding = 'utf-8')
    df7 = pd.read_csv("data/欠損値補完/FUSION/det_mean_神経障害性疼痛.csv", encoding = 'utf-8')
    df8 = pd.read_csv("data/欠損値補完/FUSION/det_mean_不明.csv", encoding = 'utf-8')
    st.markdown("①")
    st.dataframe(df6)
    csv_1 = df6.to_csv(index=False).encode('utf-8')
    st.markdown("②")
    st.dataframe(df7)
    csv_2 = df7.to_csv(index=False).encode('utf-8')
    st.markdown("③")
    st.dataframe(df8)
    csv_3 = df8.to_csv(index=False).encode('utf-8')
    st.download_button(
    label="①をダウンロード",
    data=csv_1,
    file_name='det_mean_侵害受容性疼痛.csv',
    mime='text/csv',
    )
    st.download_button(
    label="②をダウンロード",
    data=csv_2,
    file_name='det_mean_神経障害性疼痛.csv',
    mime='text/csv',
    )
    st.download_button(
    label="③をダウンロード",
    data=csv_3,
    file_name='det_mean_不明.csv',
    mime='text/csv',
    )


if home_type == "k-NN法補完":
    st.subheader('k-NN法補完', divider='rainbow')
    df9 = pd.read_csv("data/欠損値補完/FUSION/det_KNN_侵害受容性疼痛.csv", encoding = 'utf-8')
    df10 = pd.read_csv("data/欠損値補完/FUSION/det_KNN_神経障害性疼痛.csv", encoding = 'utf-8')
    df11 = pd.read_csv("data/欠損値補完/FUSION/det_KNN_不明.csv", encoding = 'utf-8')
    st.markdown("①")
    st.dataframe(df9)
    csv_1 = df9.to_csv(index=False).encode('utf-8')
    st.markdown("②")
    st.dataframe(df10)
    csv_2 = df10.to_csv(index=False).encode('utf-8')
    st.markdown("③")
    st.dataframe(df11)
    csv_3 = df11.to_csv(index=False).encode('utf-8')
    st.download_button(
    label="①をダウンロード",
    data=csv_1,
    file_name='det_KNN_侵害受容性疼痛.csv',
    mime='text/csv',
    )
    st.download_button(
    label="②をダウンロード",
    data=csv_2,
    file_name='det_KNN_神経障害性疼痛.csv',
    mime='text/csv',
    )
    st.download_button(
    label="③をダウンロード",
    data=csv_3,
    file_name='det_KNN_不明.csv',
    mime='text/csv',
    )