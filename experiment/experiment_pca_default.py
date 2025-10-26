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
from sklearn.decomposition import PCA

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
def default_pca_experiment():
    st.sidebar.header("最適化の設定")
    kernel = st.sidebar.selectbox("SVMカーネル", ["rbf", "linear", "poly", "sigmoid"], index=0, help="使用するカーネル")
    k_cv = st.sidebar.slider("StratifiedKFold の分割数 (k)", min_value=2, max_value=8, value=5, step=1)
    max_iter_svc = st.sidebar.number_input("SVC の max_iter", min_value=-1, max_value=50000, value=1500, step=100)

    st.sidebar.header("最適化の設定")
    # 主成分数をスライダーで指定
    n_components = st.sidebar.slider(
        "主成分数 (n_components)",
        min_value=2,
        max_value=20,     # ここは自動的に列数でもOKに変更可
        value=5,
        step=1
    )

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
        df1 = pd.read_csv('data/null/peindetect/questionnaire_paindetect_missing.csv', encoding = 'utf-8')
        st.markdown('#### データ')
        st.dataframe(df1)
        X_cols = df1.loc[:, "P1":"P13"].columns.tolist()
        X = df1[X_cols].copy()
        pain_col = df1.columns[1]

    elif choice_1 == '欠損値データ削除' and choice_2 == 'BS-POP':
        df1 = pd.read_csv('data/null/BSPOP/questionnaire_bspop_missing.csv', encoding = 'utf-8')
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
        df1 = pd.read_csv('data/主成分分析用/questionnaire_paindetect_median.csv', encoding = 'utf-8')
        st.markdown('#### データ')
        st.dataframe(df1)
        X_cols = df1.loc[:, "P1":"D13"].columns.tolist()
        X = df1[X_cols].copy()
        pain_col = df1.columns[1]

    elif choice_1 == '中央値補完' and choice_2 == 'BS-POP':
        df1 = pd.read_csv('data/主成分分析用/questionnaire_bspop_median.csv', encoding = 'utf-8')
        st.markdown('#### データ')
        st.dataframe(df1)
        X_cols = df1.loc[:, "D1":"D18"].columns.tolist()
        X = df1[X_cols].copy()
        pain_col = df1.columns[1]

    elif choice_1 == '中央値補完' and choice_2 == 'FUSION':
        df1 = pd.read_csv('data/主成分分析用/questionnaire_fusion_median.csv', encoding = 'utf-8')
        st.markdown('#### データ')
        st.dataframe(df1)
        X_cols = df1.loc[:, "P1":"D18"].columns.tolist()
        X = df1[X_cols].copy()
        pain_col = df1.columns[1]

    elif choice_1 == '平均値補完' and choice_2 == 'PainDITECT':
        df1 = pd.read_csv('data/主成分分析用/questionnaire_paindetect_mean.csv', encoding = 'utf-8')
        st.markdown('#### データ')
        st.dataframe(df1)
        X_cols = df1.loc[:, "P1":"D13"].columns.tolist()
        X = df1[X_cols].copy()
        pain_col = df1.columns[1]

    elif choice_1 == '平均値補完' and choice_2 == 'BS-POP':
        df1 = pd.read_csv('data/主成分分析用/questionnaire_bspop_mean.csv', encoding = 'utf-8')
        st.markdown('#### データ')
        st.dataframe(df1)
        X_cols = df1.loc[:, "D1":"D18"].columns.tolist()
        X = df1[X_cols].copy()
        pain_col = df1.columns[1]

    elif choice_1 == '平均値補完' and choice_2 == 'FUSION':
        df1 = pd.read_csv('data/主成分分析用/questionnaire_fusion_mean.csv', encoding = 'utf-8')
        st.markdown('#### データ')
        st.dataframe(df1)
        X_cols = df1.loc[:, "P1":"D18"].columns.tolist()
        X = df1[X_cols].copy()
        pain_col = df1.columns[1]

    elif choice_1 == 'k-NN法補完' and choice_2 == 'PainDITECT':
        df1 = pd.read_csv('data/主成分分析用/questionnaire_paindetect_knn.csv', encoding = 'utf-8')
        st.markdown('#### データ')
        st.dataframe(df1)
        X_cols = df1.loc[:, "P1":"D13"].columns.tolist()
        X = df1[X_cols].copy()
        pain_col = df1.columns[1]

    elif choice_1 == 'k-NN法補完' and choice_2 == 'BS-POP':
        df1 = pd.read_csv('data/主成分分析用/questionnaire_bspop_knn.csv', encoding = 'utf-8')
        st.markdown('#### データ')
        st.dataframe(df1)
        X_cols = df1.loc[:, "D1":"D18"].columns.tolist()
        X = df1[X_cols].copy()
        pain_col = df1.columns[1]


    elif choice_1 == 'k-NN法補完' and choice_2 == 'FUSION':
        df1 = pd.read_csv('data/主成分分析用/questionnaire_fusion_knn.csv', encoding = 'utf-8')
        st.markdown('#### データ')
        st.dataframe(df1)
        X_cols = df1.loc[:, "P1":"D18"].columns.tolist()
        X = df1[X_cols].copy()
        pain_col = df1.columns[1]


    options = ['する', 'しない']
    choice_4 = st.sidebar.selectbox('データの標準化', options, index = None, placeholder="選択してください")

    X_scaled = None
    feature_names = []

    # 標準化の処理（必要に応じて）
    if choice_4 == "する":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # --- 4) PCA（主成分数を指定：例 3つ） ---
    pca = PCA(n_components)

    if X_scaled is not None:
        X_pca = pca.fit_transform(X_scaled)

        # --- 5) PCA結果をデータフレーム化 ---
        pca_cols = [f"PCA{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df1.index)

        # --- 6) 疼痛種類カラム + PCA列の新しいDataFrameを作成 ---
        df_pca_final = pd.concat([df1[[pain_col]], df_pca], axis=1)

        feature_names = pca_cols  # PCA列を重み対象にする
        st.success("PCA 実行完了")

    else:
        st.info("まだ設定がされていません")

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

    #データの加工方法の指定
    options = ['欠損値削除', '中央値補完', '平均値補完', 'k-NN法補完']

    # セレクトボックスを作成し、ユーザーの選択を取得
    data_processing = st.sidebar.selectbox('欠損値補完の方法は？', options, index = None, placeholder="選択してください")

    if st.button("開始", help="実験の実行"):
        start_time = time.time()

        columns = edited_df["columns"].tolist()
        weights = edited_df["weights"].tolist()

        # --- 7) 疼痛種類で3分割 ---
        df_nociceptive = df_pca_final[df_pca_final[pain_col] == "侵害受容性疼痛"].copy()
        df_neuropathic = df_pca_final[df_pca_final[pain_col] == "神経障害性疼痛"].copy()
        df_other = df_pca_final[
            ~df_pca_final[pain_col].isin(["侵害受容性疼痛", "神経障害性疼痛"])
        ].copy()

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