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
    st.title('疼痛のデータセットの表示')
    st.markdown('#### 侵害受容性疼痛')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_1")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df1 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        st.markdown('#### データセット')
        st.dataframe(df1)

    st.markdown('#### 神経障害性疼痛')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_2")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df2 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        st.markdown('#### データセット')
        st.dataframe(df2)

    st.markdown('#### 原因不明')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_3")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df3 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        st.markdown('#### データセット')
        st.dataframe(df3)