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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
from sklearn.decomposition import PCA

MODEL_PATH = "svm_model.pkl"


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

if choice_1 == '欠損値データ削除' and choice_2 == 'PainDITECT':
    df1 = pd.read_csv('data/null/peindetect/questionnaire_paindetect_missing.csv', encoding = 'utf-8')
    st.markdown('#### データ')
    st.dataframe(df1)
    X_cols = df1.loc[:, "P1":"D13"].columns.tolist()
    X = df1[X_cols].copy()
    pain_col = df1.columns[1]

elif choice_1 == '欠損値データ削除' and choice_2 == 'BS-POP':
    df1 = pd.read_csv('data/null/BSPOP/questionnaire_bspop_missing_侵害.csv', encoding = 'utf-8')
    st.markdown('#### データ')
    st.dataframe(df1)
    X_cols = df1.loc[:, "D1":"D18"].columns.tolist()
    X = df1[X_cols].copy()
    pain_col = df1.columns[1]

elif choice_1 == '欠損値データ削除' and choice_2 == 'FUSION':
    df1 = pd.read_csv('data/null/fusion/questionnaire_fusion_missing.csv', encoding = 'utf-8')
    st.markdown('#### データ')
    st.dataframe(df1)
    X_cols = df1.loc[:, "P1":"D18"].columns.tolist()
    X = df1[X_cols].copy()
    pain_col = df1.columns[1]

elif choice_1 == '中央値補完' and choice_2 == 'PainDITECT':
    df1 = pd.read_csv('data/欠損値補完/PAINDITECT/det_median_侵害受容性疼痛_paindetect.csv', encoding = 'utf-8')
    st.markdown('#### データ')
    st.dataframe(df1)
    X_cols = df1.loc[:, "P1":"D13"].columns.tolist()
    X = df1[X_cols].copy()
    pain_col = df1.columns[1]

elif choice_1 == '中央値補完' and choice_2 == 'BS-POP':
    df1 = pd.read_csv('data/欠損値補完/BSPOP/det_median_侵害受容性疼痛_bspop.csv', encoding = 'utf-8')
    st.markdown('#### データ')
    st.dataframe(df1)
    X_cols = df1.loc[:, "D1":"D18"].columns.tolist()
    X = df1[X_cols].copy()
    pain_col = df1.columns[1]

elif choice_1 == '中央値補完' and choice_2 == 'FUSION':
    df1 = pd.read_csv('data/欠損値補完/FUSION/det_median_侵害受容性疼痛.csv', encoding = 'utf-8')
    st.markdown('#### データ')
    st.dataframe(df1)
    X_cols = df1.loc[:, "P1":"D18"].columns.tolist()
    X = df1[X_cols].copy()
    pain_col = df1.columns[1]

elif choice_1 == '平均値補完' and choice_2 == 'PainDITECT':
    df1 = pd.read_csv('data/欠損値補完/PAINDITECT/det_mean_侵害受容性疼痛_paindetect.csv', encoding = 'utf-8')
    st.markdown('#### データ')
    st.dataframe(df1)
    X_cols = df1.loc[:, "P1":"D13"].columns.tolist()
    X = df1[X_cols].copy()
    pain_col = df1.columns[1]

elif choice_1 == '平均値補完' and choice_2 == 'BS-POP':
    df1 = pd.read_csv('data/欠損値補完/BSPOP/det_mean_侵害受容性疼痛_bspop.csv', encoding = 'utf-8')
    st.markdown('#### データ')
    st.dataframe(df1)
    X_cols = df1.loc[:, "D1":"D18"].columns.tolist()
    X = df1[X_cols].copy()
    pain_col = df1.columns[1]

elif choice_1 == '平均値補完' and choice_2 == 'FUSION':
    df1 = pd.read_csv('data/欠損値補完/FUSION/det_mean_侵害受容性疼痛.csv', encoding = 'utf-8')
    st.markdown('#### データ')
    st.dataframe(df1)
    X_cols = df1.loc[:, "P1":"D18"].columns.tolist()
    X = df1[X_cols].copy()
    pain_col = df1.columns[1]

elif choice_1 == 'k-NN法補完' and choice_2 == 'PainDITECT':
    df1 = pd.read_csv('data/欠損値補完/PAINDITECT/det_KNN_侵害受容性疼痛_paindetect.csv', encoding = 'utf-8')
    st.markdown('#### データ')
    st.dataframe(df1)
    X_cols = df1.loc[:, "P1":"D13"].columns.tolist()
    X = df1[X_cols].copy()
    pain_col = df1.columns[1]

elif choice_1 == 'k-NN法補完' and choice_2 == 'BS-POP':
    df1 = pd.read_csv('data/欠損値補完/BSPOP/det_KNN_侵害受容性疼痛_bspop.csv', encoding = 'utf-8')
    st.markdown('#### データ')
    st.dataframe(df1)
    X_cols = df1.loc[:, "D1":"D18"].columns.tolist()
    X = df1[X_cols].copy()
    pain_col = df1.columns[1]


elif choice_1 == 'k-NN法補完' and choice_2 == 'FUSION':
    df1 = pd.read_csv('data/欠損値補完/FUSION/det_KNN_侵害受容性疼痛.csv', encoding = 'utf-8')
    st.markdown('#### データ')
    st.dataframe(df1)
    X_cols = df1.loc[:, "P1":"D18"].columns.tolist()
    X = df1[X_cols].copy()
    pain_col = df1.columns[1]

# if choice_2 in ["PainDITECT"]:
#     X_cols = df1.loc[:, "P1":"D13"].columns.tolist()
#     X = df1[X_cols].copy()
#     pain_col = df1.columns[1]

# if choice_2 in ["BS-POP"]:
#     X_cols = df1.loc[:, "D1":"D18"].columns.tolist()
#     X = df1[X_cols].copy()
#     pain_col = df1.columns[1]

# if choice_2 in ["FUSION"]:
#     X_cols = df1.loc[:, "P1":"D18"].columns.tolist()
#     X = df1[X_cols].copy()
#     pain_col = df1.columns[1]

# weights = []

# # セッションステートの初期化
# if "weights" not in st.session_state:
#     st.session_state.weights = {stock: 1.0 for stock in X_cols}
# if "reset" not in st.session_state:
#     st.session_state.reset = False

# # 重みの初期化
# if st.button("重みをリセット", key="weights_reset"):
#     for stock in X_cols:
#         st.session_state.weights[stock] = 1.0  # 全ての重みを初期化
#     st.session_state.reset = True

# # 動的にスライドバーを生成し、weightsに格納
# for column in X_cols:
#     if column not in st.session_state.weights:
#         st.session_state.weights[column] = 1.0
#     # セッションステートからスライダーの初期値を取得
#     default_weight = st.session_state.weights[column]
#     st.sidebar.markdown("### 重み付け")
#     weight = st.sidebar.slider(f"{column}の重み", min_value=-5.0, max_value=5.0, value=default_weight, step=0.1, key=f"slider_{column}")
#     weights.append(weight)
#     # スライダーの値をセッションステートに保存
#     st.session_state.weights[column] = weight

# # データフレームを作成
# edited_df = pd.DataFrame({"columns": X_cols, "weights": weights})

# # データフレームを表示
# st.dataframe(edited_df)

# st.markdown('#### データの標準化')
# セレクトボックスのオプションを定義
options = ['する', 'しない']

# セレクトボックスを作成し、ユーザーの選択を取得
choice_4 = st.sidebar.selectbox('データの標準化', options, index = None, placeholder="選択してください")

#データの加工方法の指定
options = ['欠損値削除', '中央値補完', '平均値補完', 'k-NN法補完']

# セレクトボックスを作成し、ユーザーの選択を取得
data_processing = st.sidebar.selectbox('欠損値補完の方法は？', options, index = None, placeholder="選択してください")


# 標準化の処理（必要に応じて）
if choice_4 == "する":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# --- 4) PCA（主成分数を指定：例 3つ） ---
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# --- 5) PCA結果をデータフレーム化 ---
pca_cols = [f"PCA{i+1}" for i in range(3)]
df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df1.index)

# --- 6) 疼痛種類カラム + PCA列の新しいDataFrameを作成 ---
df_pca_final = pd.concat([df1[[pain_col]], df_pca], axis=1)

feature_names = pca_cols  # PCA列を重み対象にする

# セッションステート初期化
if "weights" not in st.session_state:
    st.session_state.weights = {col: 1.0 for col in feature_names}
if "reset" not in st.session_state:
    st.session_state.reset = False

# 重みリセットボタン
if st.button("重みをリセット", key="weights_reset"):
    for col in feature_names:
        st.session_state.weights[col] = 1.0
    st.session_state.reset = True
    st.success("全ての重みを1.0にリセットしました")

# サイドバータイトル
st.sidebar.markdown("### 重み付け（PCA列）")

# スライダー生成
weights = []
for col in feature_names:
    if col not in st.session_state.weights:
        st.session_state.weights[col] = 1.0
    default_weight = st.session_state.weights[col]
    weight = st.sidebar.slider(
        f"{col} の重み",
        min_value=-5.0, max_value=5.0,
        value=default_weight, step=0.1,
        key=f"slider_{col}"
    )
    st.session_state.weights[col] = weight
    weights.append(weight)

# 重み確認用データフレーム
edited_df = pd.DataFrame({"columns": feature_names, "weights": weights})
st.write("現在の重み（PCA列）")
st.dataframe(edited_df, use_container_width=True)

if st.button("開始", help="実験の実行"):
    columns = edited_df["columns"].tolist()
    weights = edited_df["weights"].tolist()

    # # 標準化の処理（必要に応じて）
    # if choice_4 == "する":
    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X)

    # # --- 4) PCA（主成分数を指定：例 3つ） ---
    # pca = PCA(n_components=3)
    # X_pca = pca.fit_transform(X_scaled)

    # # --- 5) PCA結果をデータフレーム化 ---
    # pca_cols = [f"PCA{i+1}" for i in range(3)]
    # df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df1.index)

    # # --- 6) 疼痛種類カラム + PCA列の新しいDataFrameを作成 ---
    # df_pca_final = pd.concat([df1[[pain_col]], df_pca], axis=1)

    # --- 7) 疼痛種類で3分割 ---
    df_nociceptive = df_pca_final[df_pca_final[pain_col] == "侵害受容性疼痛"].copy()
    df_neuropathic = df_pca_final[df_pca_final[pain_col] == "神経障害性疼痛"].copy()
    df_other = df_pca_final[
        ~df_pca_final[pain_col].isin(["侵害受容性疼痛", "神経障害性疼痛"])
    ].copy()

    # # 重みの初期化
    # if st.button("重みをリセット", key="weights_reset"):
    #     for stock in X_cols:
    #         st.session_state.weights[stock] = 1.0  # 全ての重みを初期化
    #     st.session_state.reset = True

    # # 動的にスライドバーを生成し、weightsに格納
    # for column in X_cols:
    #     if column not in st.session_state.weights:
    #         st.session_state.weights[column] = 1.0
    #     # セッションステートからスライダーの初期値を取得
    #     default_weight = st.session_state.weights[column]
    #     st.sidebar.markdown("### 重み付け")
    #     weight = st.sidebar.slider(f"{column}の重み", min_value=-5.0, max_value=5.0, value=default_weight, step=0.1, key=f"slider_{column}")
    #     weights.append(weight)
    #     # スライダーの値をセッションステートに保存
    #     st.session_state.weights[column] = weight
    
    # データの指定
    df_nociceptive_train = df_nociceptive[columns]
    df_neuronociceptive_train = df_neuropathic[columns]
    df_unknown_train = df_other[columns]

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

    initial_weights = np.random.randint(-5, 5, datas.shape[1]).astype(float)

    # 重みをかける関数
    def apply_weights(datas, weights_change):
        return datas * weights_change

    # 指定された重みで交差検証精度を返す関数
    def evaluate(weights_change, datas, labels, C, k=5, return_best_split=False):
        X_weighted = apply_weights(datas, weights_change)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = []

        best_fold_score = 0
        best_X_val, best_y_val, best_pred = None, None, None

        for train_index, val_index in skf.split(X_weighted, labels):
            X_train, X_val = X_weighted[train_index], X_weighted[val_index]
            y_train, y_val = labels[train_index], labels[val_index]

            model = SVC(C=C, kernel='linear', max_iter=1500)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = np.mean(y_pred == y_val)
            scores.append(acc)

            # 評価指標が最高のfoldを保存
            if return_best_split and acc > best_fold_score:
                best_fold_score = acc
                best_X_val = X_val
                best_y_val = y_val
                best_pred = y_pred

        if return_best_split:
                return np.mean(scores), best_X_val, best_y_val, best_pred
        else:
            return np.mean(scores)

    # 山登り法（1つのCに対して最適な重みを探索）
    def hill_climbing(datas, labels, C, max_iter_1=1000, step_size=0.01):
        n_features = datas.shape[1]
        weights_change = np.ones(n_features).astype(float)
        # weights_change = initial_weights.copy()  # 外から渡された固定の初期重み
        st.write("✅ 初期重み:" + str([int(w) for w in weights_change]))

        best_score, best_X_val, best_y_val, best_pred = evaluate(weights_change, datas, labels, C
        , return_best_split=True)
        best_weights = weights_change.copy()


        # Streamlitの進捗バーとスコア表示
        hill_bar = st.progress(0)
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
                        trial_weights, datas, labels, C, return_best_split=True
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
            percent = int((i + 1) / max_iter_1 * 100)
            hill_bar.progress(percent, text=f"進捗状況{percent}%")

        return best_weights, max(score_history), best_X_val, best_y_val, best_pred, score_history

    C_values = [1, 0.1]
    best_score = 0
    best_C = None
    best_weights = None
    best_X_val = best_y_val = best_pred = None

    # Cのグリッドサーチ（外側ループ）
    for C in C_values:
        weights_change, score, X_val_tmp, y_val_tmp, pred_tmp, score_history = hill_climbing(datas, labels, C, max_iter_1=1000, step_size=0.01)
        st.write(f"→ C={C} で得られたスコア: {score:.4f}")
        # グラフ描画
        fig, ax = plt.subplots()
        ax.plot(score_history)
        ax.set_title("Score progression by Hill Climbing")
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        st.pyplot(fig)

        if score > best_score:
            best_score = score
            best_C = C
            best_weights = weights_change
            best_X_val = X_val_tmp
            best_y_val = y_val_tmp
            best_pred = pred_tmp

    # 最終モデルを学習＆保存
    X_weighted_final = apply_weights(datas, best_weights)
    final_model = SVC(C=best_C, kernel='linear', max_iter=1500)
    final_model.fit(X_weighted_final, labels)
    joblib.dump(final_model, MODEL_PATH)

    # データフレームを作成
    best_weights_df = pd.DataFrame({"columns": feature_names, "weights": best_weights})

    # 結果表示
    st.write("✅ 最適なC:", best_C)
    st.write("✅ 最適な重み:")
    st.dataframe(best_weights_df)
    st.write("✅ 最終スコア:", best_score)

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