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
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def show_pca_analysis(df, start_col='P1', end_col='D13', pain_col=None):
    # --- 列指定確認 ---
    if pain_col is None:
        pain_col = df.columns[1]

    # ==== 特徴量抽出 ====
    X_full = df.loc[:, start_col:end_col].copy()
    valid_idx = X_full.dropna().index
    X = X_full.loc[valid_idx]
    labels = df.loc[valid_idx, pain_col].astype(str).values

    lab_noci = "侵害受容性疼痛"
    lab_neur = "神経障害性疼痛"

    st.subheader("相関ヒートマップ")
    corr = X.corr()
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax1)
    ax1.set_title("Correlation Heatmap")
    st.pyplot(fig1)

    # ==== 標準化 ====
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==== PCA ====
    st.subheader("PCA実施")
    max_n = min(30, X.shape[1])  # 主成分の最大数（必要なら増やしてOK）
    n_components = st.slider("主成分数 (n_components)", min_value=2, max_value=max_n, value=min(15, max_n), step=1)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)  # ← fit + transform
    pca_cols = [f"PC{i+1}" for i in range(n_components)]

    # ==== 主成分負荷量（ロードings） ====
    st.subheader("各質問項目の主成分負荷量（元の質問項目 × 主成分）")
    loadings = pd.DataFrame(pca.components_.T, columns=pca_cols, index=X.columns)
    fig2, ax2 = plt.subplots(figsize=(max(10, n_components*0.6), 10))
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0, ax=ax2)
    ax2.set_title("Principal Component Loadings")
    ax2.set_xlabel("PCA")
    ax2.set_ylabel("questions")
    st.pyplot(fig2)

    # ==== 散布図（任意の2主成分を比較） ====
    st.subheader("PCA スコア散布図")
    st.markdown("""
    - ⚫︎：侵害受容性疼痛(nociceptive)
    - ▲：神経障害性疼痛(neuropathic)
    - ◼︎：不明(unknown)
    """)
    # 0始まりインデックスを使う（表示はPC番号）
    pc_x_name = st.selectbox("横軸", pca_cols, index=min(0, n_components-1))
    pc_y_name = st.selectbox("縦軸", pca_cols, index=min(1, n_components-1))
    pc_x = pca_cols.index(pc_x_name)
    pc_y = pca_cols.index(pc_y_name)

    mask_noci = (labels == lab_noci)
    mask_neur = (labels == lab_neur)
    mask_other = ~(mask_noci | mask_neur)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(X_pca[mask_noci, pc_x], X_pca[mask_noci, pc_y], label="nociceptive", alpha=0.8, marker='o')
    ax3.scatter(X_pca[mask_neur, pc_x], X_pca[mask_neur, pc_y], label="neuropathic", alpha=0.8, marker='^')
    ax3.scatter(X_pca[mask_other, pc_x], X_pca[mask_other, pc_y], label="unknown", alpha=0.6, marker='s')
    ax3.set_xlabel(pc_x_name)
    ax3.set_ylabel(pc_y_name)
    ax3.set_title(f"PCA-({pc_x_name} vs {pc_y_name})")
    ax3.legend(loc="best", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig3)

    # ==== 補足情報 ====
    st.markdown("**寄与率**")
    cumexp = np.cumsum(pca.explained_variance_ratio_)
    st.write(pd.DataFrame({
        "PC": pca_cols,
        "寄与率": np.round(pca.explained_variance_ratio_, 4),
        "累積寄与率": np.round(cumexp, 4),
    }))



st.title('データ分析')

# セレクトボックスのオプションを定義
options = ['欠損値データ削除', '中央値補完', '平均値補完', 'k-NN法補完']

options_1 = ['標準化','相関係数']

# ラジオボタンを表示
home_type = st.sidebar.radio("選んでください", ["データ分布", "データ変換", "主成分分析"])

# ファイル読み込みと処理
if home_type == "データ分布":
    # セレクトボックスを作成し、ユーザーの選択を取得
    choice_1 = st.selectbox('データ分布', options, index = None, placeholder="選択してください")
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

if home_type == "データ変換":
    # セレクトボックスを作成し、ユーザーの選択を取得
    choice_2 = st.selectbox('データ変換', options_1, index = None, placeholder="選択してください")
    if choice_2 == '標準化':
        st.subheader("📂 アップロード")
        uploaded_file = st.file_uploader("標準化したいCSVファイルをアップロードしてください", type=["csv"])

        if uploaded_file is not None:
            # CSV読み込み
            df = pd.read_csv(uploaded_file)
            # 数値データの標準化
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
            scaler = StandardScaler()
            df_standardized = df.copy()
            df_standardized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            # データ表示
            st.subheader("📊 標準化後のデータ")
            st.dataframe(df_standardized)
    elif choice_2 == '相関係数':
        st.title("偏相関ヒートマップ可視化")
        # --- ファイルアップロード ---
        uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv", key="unique_key_all")

        if uploaded_file:
            # === 読み込み & 0,1列目を削除 ===
            df = pd.read_csv(uploaded_file, encoding="utf-8")
            if df.shape[1] < 3:
                st.error("列数が不足しています（最低3列以上必要）。")
                st.stop()
            df = df.iloc[:, 2:]  # 0,1列目を削除

            # 数値列のみ取り出し（テキスト列等は除外）
            df_num = df.select_dtypes(include=[np.number]).copy()

            # 全欠損列・定数列は除外（回帰できないため）
            nunique = df_num.nunique(dropna=True)
            usable_cols = nunique[nunique > 1].index.tolist()
            df_num = df_num[usable_cols]

            if df_num.shape[1] < 2:
                st.error("偏相関を計算できる数値列（かつ定数でない列）が2つ以上必要です。")
                st.stop()

            st.write("🧾 0,1列削除後の数値データプレビュー：")
            st.dataframe(df_num.head())

            # === 偏相関を計算する関数（target1 と target2 の他を制御変数に） ===
            def partial_corr_pairwise(df_, t1, t2, controls):
                cols_needed = [t1, t2] + controls
                d = df_[cols_needed].dropna()
                # サンプルが少ない/制御変数が多すぎると不安定になるため簡易チェック
                if d.shape[0] < 3 or len(controls) == 0:
                    # controls が空のときは通常の相関
                    if len(controls) == 0 and d.shape[0] >= 2:
                        return np.corrcoef(d[t1], d[t2])[0, 1]
                    return np.nan

                # 線形回帰で残差を取り出す
                try:
                    m1 = LinearRegression().fit(d[controls], d[t1])
                    r1 = d[t1] - m1.predict(d[controls])

                    m2 = LinearRegression().fit(d[controls], d[t2])
                    r2 = d[t2] - m2.predict(d[controls])

                    return np.corrcoef(r1, r2)[0, 1]
                except Exception:
                    return np.nan

            # === 偏相関行列の作成（全カラム対象） ===
            vars_all = df_num.columns.tolist()
            n = len(vars_all)
            pcorr = pd.DataFrame(np.eye(n), index=vars_all, columns=vars_all, dtype=float)

            st.write("⏳ 偏相関を計算中...")
            progress = st.progress(0.0)
            total_pairs = n * (n - 1) / 2
            done = 0

            for i in range(n):
                for j in range(i + 1, n):
                    v1, v2 = vars_all[i], vars_all[j]
                    controls = [v for v in vars_all if v not in (v1, v2)]
                    r = partial_corr_pairwise(df_num, v1, v2, controls)
                    pcorr.loc[v1, v2] = r
                    pcorr.loc[v2, v1] = r
                    done += 1
                    progress.progress(done / total_pairs if total_pairs > 0 else 1.0)

            progress.empty()

            # === ヒートマップ描画 ===
            st.subheader("🔥 偏相関ヒートマップ（全数値カラム）")
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(
                pcorr.astype(float),
                vmin=-1, vmax=1,
                annot=True, fmt=".2f",
                cmap=sns.color_palette("coolwarm", 100)
                # square=True
            )
            ax.set_title("Partial Correlation Matrix (after dropping first two columns)", fontsize=14)
            st.pyplot(fig)

            # === 偏相関行列のCSVダウンロード ===
            csv_data = pcorr.round(4).to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                label="📥 偏相関行列をCSVでダウンロード",
                data=csv_data,
                file_name="partial_correlation_matrix.csv",
                mime="text/csv"
            )

            # === 表形式の確認 ===
            with st.expander("📋 偏相関行列（表形式で確認）"):
                st.dataframe(pcorr.round(3))

if home_type == "主成分分析":
    st.header("主成分分析（PCA）")

    choice = st.selectbox("どのデータにする？", ["欠損値削除(FUSION)", "欠損値削除(PainDETECT)","欠損値削除(BS-POP)"])

    if choice == "欠損値削除(FUSION)":
        df = pd.read_csv("data/null/fusion/questionnaire_fusion_missing.csv")
        show_pca_analysis(df, start_col='P1', end_col='D18', pain_col=None)

    elif choice == "欠損値削除(PainDETECT)":
        df = pd.read_csv("data/null/peindetect/questionnaire_paindetect_missing.csv")
        show_pca_analysis(df, start_col='P1', end_col='P13', pain_col=None)

    elif choice == "欠損値削除(BS-POP)":
        df = pd.read_csv("data/null/BSPOP/questionnaire_bspop_missing.csv")
        show_pca_analysis(df, start_col='D1', end_col='D18', pain_col=None)


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