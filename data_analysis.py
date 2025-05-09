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
    st.title('データ分析')
    st.markdown('偏相関係数の評価を表示')
    st.markdown('#### PainDETECT')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_1")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df8 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        # データフレーム内の2つの変数 (target1 と target2) の間で、他の変数 (control_vars) の影響を取り除いた偏相関係数を計算
        def calculate_partial_correlation(df, target1, target2, control_vars):
            # 回帰モデルの作成と実測値から予測値の残差の計算
            model1 = LinearRegression().fit(df[control_vars], df[target1])
            residuals_target1 = df[target1] - model1.predict(df[control_vars])

            model2 = LinearRegression().fit(df[control_vars], df[target2])
            residuals_target2 = df[target2] - model2.predict(df[control_vars])

            # 残差間の相関を計算
            partial_corr = np.corrcoef(residuals_target1, residuals_target2)[0, 1]
            return partial_corr

        # 新しい変数リストの作成（P1からP13とD1からD18）
        variables = [f'P{i}' for i in range(1, 14)]
        partial_corr_matrix = pd.DataFrame(index=variables, columns=variables)

        # ループを使って全てのペアで偏相関を計算
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                target1 = variables[i]
                target2 = variables[j]
                
                # target1とtarget2を除いた他の変数を制御変数とする
                control_vars = [v for v in variables if v != target1 and v != target2]
                
                # 各ペアの偏相関を計算
                partial_corr = calculate_partial_correlation(df8, target1, target2, control_vars)
                
                # 行列に対称的に格納
                partial_corr_matrix.loc[target1, target2] = partial_corr
                partial_corr_matrix.loc[target2, target1] = partial_corr

        # 対角成分に1を設定（自己相関）
        np.fill_diagonal(partial_corr_matrix.values, 1)

        pd.set_option('display.max_columns', 50)

        # 数値を小数第3位まで丸める
        pd.options.display.float_format = '{:.2f}'.format

        # 現在の日時を取得してファイル名に追加
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_path = f'/Users/iwasho_0225/Desktop/workspace/pain_experiment/相関係数/PainDETECT/partial_{timestamp}.csv'

        # CSVファイルとして出力
        partial_corr_matrix.to_csv(output_path, float_format="%.3f")

        # 完了メッセージの表示
        st.success(f"CSVファイルが '{output_path}' に保存されました。")

    st.markdown('#### BS-POP')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_2")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df8 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        # データフレーム内の2つの変数 (target1 と target2) の間で、他の変数 (control_vars) の影響を取り除いた偏相関係数を計算
        def calculate_partial_correlation(df, target1, target2, control_vars):
            # 回帰モデルの作成と実測値から予測値の残差の計算
            model1 = LinearRegression().fit(df[control_vars], df[target1])
            residuals_target1 = df[target1] - model1.predict(df[control_vars])

            model2 = LinearRegression().fit(df[control_vars], df[target2])
            residuals_target2 = df[target2] - model2.predict(df[control_vars])

            # 残差間の相関を計算
            partial_corr = np.corrcoef(residuals_target1, residuals_target2)[0, 1]
            return partial_corr

        # 新しい変数リストの作成（P1からP13とD1からD18）
        variables = [f'D{i}' for i in range(1, 19)]
        partial_corr_matrix = pd.DataFrame(index=variables, columns=variables)

        # ループを使って全てのペアで偏相関を計算
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                target1 = variables[i]
                target2 = variables[j]
                
                # target1とtarget2を除いた他の変数を制御変数とする
                control_vars = [v for v in variables if v != target1 and v != target2]
                
                # 各ペアの偏相関を計算
                partial_corr = calculate_partial_correlation(df8, target1, target2, control_vars)
                
                # 行列に対称的に格納
                partial_corr_matrix.loc[target1, target2] = partial_corr
                partial_corr_matrix.loc[target2, target1] = partial_corr

        # 対角成分に1を設定（自己相関）
        np.fill_diagonal(partial_corr_matrix.values, 1)

        pd.set_option('display.max_columns', 50)

        # 数値を小数第3位まで丸める
        pd.options.display.float_format = '{:.2f}'.format

        # 現在の日時を取得してファイル名に追加
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_path = f'/Users/iwasho_0225/Desktop/workspace/pain_experiment/相関係数/BSPOP/partial_{timestamp}.csv'

        # CSVファイルとして出力
        partial_corr_matrix.to_csv(output_path, float_format="%.3f")

        # 完了メッセージの表示
        st.success(f"CSVファイルが '{output_path}' に保存されました。")

    st.markdown('#### FUSION')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_3")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df8 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        def calculate_partial_correlation(df, target1, target2, control_vars):
            model1 = LinearRegression().fit(df[control_vars], df[target1])
            residuals_target1 = df[target1] - model1.predict(df[control_vars])

            model2 = LinearRegression().fit(df[control_vars], df[target2])
            residuals_target2 = df[target2] - model2.predict(df[control_vars])

            partial_corr = np.corrcoef(residuals_target1, residuals_target2)[0, 1]
            return partial_corr

        # 新しい変数リストの作成（P1からP13とD1からD18）
        variables = [f'P{i}' for i in range(1, 14)] + [f'D{i}' for i in range(1, 19)]
        partial_corr_matrix = pd.DataFrame(index=variables, columns=variables)

        # ループを使って全てのペアで偏相関を計算
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                target1 = variables[i]
                target2 = variables[j]
                
                # target1とtarget2を除いた他の変数を制御変数とする
                control_vars = [v for v in variables if v != target1 and v != target2]
                
                # 各ペアの偏相関を計算
                partial_corr = calculate_partial_correlation(df8, target1, target2, control_vars)
                
                # 行列に対称的に格納
                partial_corr_matrix.loc[target1, target2] = partial_corr
                partial_corr_matrix.loc[target2, target1] = partial_corr

        # 対角成分に1を設定（自己相関）
        np.fill_diagonal(partial_corr_matrix.values, 1)

        pd.set_option('display.max_columns', 50)

        # 数値を小数第3位まで丸める
        pd.options.display.float_format = '{:.2f}'.format

        # 現在の日時を取得してファイル名に追加
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_path = f'/Users/iwasho_0225/Desktop/workspace/pain_experiment/相関係数/FUSION/partial_{timestamp}.csv'

        # CSVファイルとして出力
        partial_corr_matrix.to_csv(output_path, float_format="%.3f")

        # 完了メッセージの表示
        st.success(f"CSVファイルが '{output_path}' に保存されました。")

    st.markdown('#### 相関係数のCSVファイル参照')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_4")

    if uploaded_file:
        # CSVファイルの読み込み
        df = pd.read_csv(uploaded_file)
        
        # データを小数第3位まで丸める
        df = df.round(3)
        
        # データフレームとして出力
        st.dataframe(df)