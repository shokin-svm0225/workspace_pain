import streamlit as st
import itertools
import plotly.express as px
import matplotlib.pyplot as plt
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

st.title('データ分析')

# セレクトボックスのオプションを定義
options = ['欠損値データ削除', '中央値補完', '平均値補完', 'k-NN法補完']

# セレクトボックスを作成し、ユーザーの選択を取得
choice_1 = st.sidebar.selectbox('データ分析', options, index = None, placeholder="選択してください")

# ファイル読み込みと処理
if choice_1:
    if choice_1 == '欠損値データ削除':
        df1 = pd.read_csv('data/null/fusion/questionnaire_fusion_missing_侵害.csv')
        df2 = pd.read_csv('data/null/fusion/questionnaire_fusion_missing_神経.csv')
        df3 = pd.read_csv('data/null/fusion/questionnaire_fusion_missing_不明.csv')
    elif choice_1 == '中央値補完':
        df1 = pd.read_csv('data/欠損値補完/FUSION/det_median_侵害受容性疼痛.csv')
        df2 = pd.read_csv('data/欠損値補完/FUSION/det_median_神経障害性疼痛.csv')
        df3 = pd.read_csv('data/欠損値補完/FUSION/det_median_不明.csv')
    elif choice_1 == '平均値補完':
        df1 = pd.read_csv('data/欠損値補完/FUSION/det_mean_侵害受容性疼痛.csv')
        df2 = pd.read_csv('data/欠損値補完/FUSION/det_mean_神経障害性疼痛.csv')
        df3 = pd.read_csv('data/欠損値補完/FUSION/det_mean_不明.csv')
    elif choice_1 == 'k-NN法補完':
        df1 = pd.read_csv('data/欠損値補完/FUSION/det_KNN_侵害受容性疼痛.csv')
        df2 = pd.read_csv('data/欠損値補完/FUSION/det_KNN_神経障害性疼痛.csv')
        df3 = pd.read_csv('data/欠損値補完/FUSION/det_KNN_不明.csv')

    # 質問項目の抽出
    question_cols = [col for col in df1.columns if col.startswith('P') or col.startswith('D')]

    # 度数分布を計算する関数
    def calculate_value_counts(df, columns):
        return pd.DataFrame({col: df[col].value_counts().sort_index() for col in columns}).fillna(0)

    counts_df1 = calculate_value_counts(df1, question_cols)
    counts_df2 = calculate_value_counts(df2, question_cols)
    counts_df3 = calculate_value_counts(df3, question_cols)

    score_range = sorted(set(counts_df1.index).union(counts_df2.index).union(counts_df3.index))

    # 2列表示（matplotlibのグリッドで配置）
    n_cols = 2
    n_rows = int(np.ceil(len(question_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    legend_labels = []  # 凡例ラベル管理

    for i, col in enumerate(question_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r][c] if n_cols > 1 else axes[r]
        s_vals = counts_df1[col].reindex(score_range, fill_value=0)
        k_vals = counts_df2[col].reindex(score_range, fill_value=0)
        f_vals = counts_df3[col].reindex(score_range, fill_value=0)

        bar1 = ax.bar(score_range, s_vals, label='Nociceptive Pain', color='navy')
        bar2 = ax.bar(score_range, k_vals, bottom=s_vals, label='Neuropathic Pain', color='skyblue')
        bar3 = ax.bar(score_range, f_vals, bottom=s_vals + k_vals, label='Unknown', color='red')

        if i == 0:
            legend_labels = [bar1[0], bar2[0], bar3[0]]  # 最初のグラフからのみ取得

        ax.set_title(f'{col}')
        ax.set_xlabel('score')
        ax.set_ylabel('people')
        ax.set_xticks(score_range)

    fig.suptitle('Score_distribution[P1-D18]', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.legend(legend_labels, ['Nociceptive Pain', 'Neuropathic Pain', 'Unknown'], loc='upper right')

    st.pyplot(fig)





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