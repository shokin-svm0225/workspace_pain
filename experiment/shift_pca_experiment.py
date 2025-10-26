
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import joblib

import streamlit as st
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.decomposition import PCA

# =========================
# ユーティリティ
# =========================
def parse_num_list(s, dtype=float):
    """
    '0.1, 1, 10' -> [0.1, 1.0, 10.0] のように変換する。
    空や不正な値は無視。dtype は float か int を想定。
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

# === 共通関数群 ===
def apply_weights(datas, weights_change):
    return datas * weights_change

def build_svc(kernel, C, gamma=None, degree=None, coef0=None, max_iter=1500):
    # SVC のパラメータは kernel により無視されるものがあるが、渡しても問題はない。
    params = dict(C=C, kernel=kernel, max_iter=max_iter)
    if gamma is not None:
        params["gamma"] = gamma
    if degree is not None:
        params["degree"] = degree
    if coef0 is not None:
        params["coef0"] = coef0
    return SVC(**params)

def evaluate(weights_change, datas, labels, C, kernel, gamma=None, degree=None, coef0=None, k=5, return_best_split=False, max_iter_svc=1500):
    X_weighted = apply_weights(datas, weights_change)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    best_fold_score = -np.inf
    best_X_val, best_y_val, best_pred = None, None, None

    for train_index, val_index in skf.split(X_weighted, labels):
        X_train, X_val = X_weighted[train_index], X_weighted[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        model = build_svc(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0, max_iter=max_iter_svc)
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

#山登り法(2方向)
def hill_climbing_1(datas, labels, C, kernel, gamma, degree, coef0, weights_init, max_iter_1=1000, step_size=0.01, k=5, max_iter_svc=1500):
    n_features = datas.shape[1]
    if weights_init is None:
        weights_init = np.ones(n_features, dtype=float)
    elif isinstance(weights_init, list):
        weights_init = np.asarray(weights_init, dtype=float)
    weights_change = weights_init.copy()

    best_score, best_X_val, best_y_val, best_pred = evaluate(
        weights_change, datas, labels, C, kernel, gamma, degree, coef0, k=k, return_best_split=True, max_iter_svc=max_iter_svc
    )
    best_weights = weights_change.copy()
    score_history = [best_score]

    global_best_score = best_score
    global_best_weights = best_weights.copy()
    global_best_pack = (best_X_val, best_y_val, best_pred)

    for _ in range(max_iter_1):
        step_best_score = -np.inf
        candidates = []
        trial_configs = []

        for idx in range(n_features):
            for delta in [-step_size, step_size]:
                trial_weights = weights_change.copy().astype(float)
                trial_weights[idx] += delta
                trial_configs.append((trial_weights, idx, delta))

        results = []
        for tw, _, _ in trial_configs:
            results.append(
                evaluate(tw, datas, labels, C, kernel, gamma, degree, coef0,
                        k=k, return_best_split=True, max_iter_svc=max_iter_svc)
            )

        for i in range(len(trial_configs)):
            trial_weights, _, _ = trial_configs[i]
            score, X_val_tmp, y_val_tmp, pred_tmp = results[i]

            if score > step_best_score:
                step_best_score = score
                candidates = [(trial_weights.copy(), X_val_tmp, y_val_tmp, pred_tmp)]
            elif score == step_best_score:
                candidates.append((trial_weights.copy(), X_val_tmp, y_val_tmp, pred_tmp))

        # ✅ スコアが同じ候補からランダムに1つを選ぶ
        selected_weights, selected_X_val, selected_y_val, selected_pred = random.choice(candidates)
        weights_change = selected_weights
        best_weights = weights_change.copy()
        best_score = step_best_score
        score_history.append(best_score)

        # ★ 追加: グローバルベストを改善時のみ更新（返却の整合性用）
        if best_score >= global_best_score:
            global_best_score = best_score
            global_best_weights = best_weights.copy()
            global_best_pack = (selected_X_val, selected_y_val, selected_pred)

    # ★ 変更: 返り値は“グローバルベスト”に統一
    return global_best_weights, global_best_score, global_best_pack[0], global_best_pack[1], global_best_pack[2], score_history

#山登り法(3方向)
def hill_climbing_2(datas, labels, C, kernel, gamma, degree, coef0, weights_init, max_iter_1=1000, step_size=0.01, k=5, max_iter_svc=1500):
    n_features = datas.shape[1]
    if weights_init is None:
        weights_init = np.ones(n_features, dtype=float)
    elif isinstance(weights_init, list):
        weights_init = np.asarray(weights_init, dtype=float)
    weights_change = weights_init.copy()

    # 初期評価
    best_score, best_X_val, best_y_val, best_pred = evaluate(
        weights_change, datas, labels, C, kernel, gamma, degree, coef0,
        k=k, return_best_split=True, max_iter_svc=max_iter_svc
    )
    best_weights = weights_change.copy()
    score_history = [best_score]

    # グローバルベスト追跡
    global_best_score = best_score
    global_best_weights = best_weights.copy()
    global_best_pack = (best_X_val, best_y_val, best_pred)

    # === 3方向（-step, 0, +step）ヒルクライミング ===
    for i in range(max_iter_1):
        step_best_score = best_score
        step_best_weights = best_weights.copy()
        step_best_pack = (best_X_val, best_y_val, best_pred)

        for idx in range(n_features):
            for delta in (-step_size, 0.0, step_size):
                trial_weights = best_weights.copy()
                trial_weights[idx] += delta

                if delta == 0.0:
                    score, Xv, yv, pred = best_score, best_X_val, best_y_val, best_pred
                else:
                    score, Xv, yv, pred = evaluate(
                        trial_weights, datas, labels, C, kernel, gamma, degree, coef0,
                        k=k, return_best_split=True, max_iter_svc=max_iter_svc
                    )

                if score > step_best_score:
                    step_best_score = score
                    step_best_weights = trial_weights
                    step_best_pack = (Xv, yv, pred)

        # 改善があれば更新、なければ早期終了
        if step_best_score > best_score:
            best_score = step_best_score
            best_weights = step_best_weights
            best_X_val, best_y_val, best_pred = step_best_pack
            score_history.append(best_score)

            if best_score >= global_best_score:
                global_best_score = best_score
                global_best_weights = best_weights.copy()
                global_best_pack = step_best_pack
        else:
            break

    return global_best_weights,global_best_score,global_best_pack[0],global_best_pack[1],global_best_pack[2],score_history



def run_hill_climbing_1(step_size, kernel, gamma, degree, coef0, C, datas, labels, weights_init, max_iter_hc=1000, k=5, max_iter_svc=1500):
    n_features = datas.shape[1]
    if weights_init is None:
        weights_init = np.ones(n_features, dtype=float)
    elif isinstance(weights_init, list):
        weights_init = np.asarray(weights_init, dtype=float)

    weights_best, score, X_val_tmp, y_val_tmp, pred_tmp, score_history = hill_climbing_1(
        datas, labels, C, kernel, gamma, degree, coef0, weights_init, max_iter_1=max_iter_hc, step_size=step_size, k=k, max_iter_svc=max_iter_svc
    )
    return {
        "step_size": step_size,
        "kernel": kernel,
        "gamma": gamma,
        "degree": degree,
        "coef0": coef0,
        "C": C,
        "score": score,
        "weights": [float(f"{w:.2f}") for w in weights_best],
        "weights_raw": np.asarray(weights_best, dtype=float).tolist(),
        "score_history": score_history,
        "X_val": X_val_tmp,
        "y_val": y_val_tmp,
        "pred": pred_tmp,
    }

def run_hill_climbing_2(step_size, kernel, gamma, degree, coef0, C, datas, labels, weights_init, max_iter_hc=1000, k=5, max_iter_svc=1500):
    n_features = datas.shape[1]
    if weights_init is None:
        weights_init = np.ones(n_features, dtype=float)
    elif isinstance(weights_init, list):
        weights_init = np.asarray(weights_init, dtype=float)
    
    weights_best, score, X_val_tmp, y_val_tmp, pred_tmp, score_history = hill_climbing_2(
        datas, labels, C, kernel, gamma, degree, coef0, weights_init, max_iter_1=max_iter_hc, step_size=step_size, k=k, max_iter_svc=max_iter_svc
    )
    return {
        "step_size": step_size,
        "kernel": kernel,
        "gamma": gamma,
        "degree": degree,
        "coef0": coef0,
        "C": C,
        "score": score,
        "weights": [float(f"{w:.2f}") for w in weights_best],
        "weights_raw": np.asarray(weights_best, dtype=float).tolist(),
        "score_history": score_history,
        "X_val": X_val_tmp,
        "y_val": y_val_tmp,
        "pred": pred_tmp,
    }

# =========================
# UI & メイン処理
# =========================
def run_shift_pca_experiment():
    hill_type = st.sidebar.radio("山登り法の選択",["2方向", "3方向"], index=0)

    if hill_type == "2方向":
        run_hill = run_hill_climbing_1

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('山登り法(2方向)', divider='rainbow')
                st.markdown("""
                - グローバルベスト  
                各特徴量ごとに「+ε/-εの二方向」で正答率を出し、2×n(特徴量)通りの中で一番良い方向に更新していく
                """)
            with col2:
                st.code(
                    "重み = [1, 1, 1, 1, 1]   ← 初期状態\n"
                    "↓\n"
                    "各特徴量について  重み + [-ε, +ε] の2通りを試す\n"
                    "→ スコアが最も良い重みを記録\n"
                    "↓\n"
                    "全特徴量を一巡したら一番良かった重みに更新\n"
                    "↓\n"
                    "これを max_iter 回繰り返す",
                    language="text"
                )

    elif hill_type == "3方向":
        run_hill = run_hill_climbing_2
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                # with st.container(border=True):
                st.subheader('山登り法(3方向)', divider='rainbow')
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

    # ===== サイドバー: アルゴリズム設定 =====
    st.sidebar.header("最適化の設定")
    kernel = st.sidebar.selectbox("SVMカーネル", ["linear", "rbf", "poly", "sigmoid"], index=0, help="使用するカーネル")

    step_sizes_str = st.sidebar.text_input("ステップ幅", value="0.01", help="例: 0.005, 0.01, 0.02")
    step_sizes = parse_num_list(step_sizes_str, float)
    if not step_sizes:
        st.sidebar.warning("ステップ幅の候補が空です。デフォルト 0.01 を使用します。")
        step_sizes = [0.01]

    max_iter_hc = st.sidebar.number_input("山登り法の反復回数 (max_iter)", min_value=1, max_value=5000, value=1000, step=50)
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

    # ===== ここからデータ選択（元コード維持） =====

    st.sidebar.header("データセット設定")

    options = ['欠損値データ削除', '中央値補完', '平均値補完', 'k-NN法補完']
    choice_1 = st.sidebar.selectbox('欠損値の対応', options, index = None, placeholder="選択してください")

    options = ['PainDITECT', 'BS-POP', 'FUSION']
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
    pca = PCA(n_components, svd_solver="full")

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

    st.markdown('#### 重みづけの指定')

    # --- 初期重み設定 ---
    st.sidebar.header("初期設定")

    n_features = len(feature_names)
    if n_features == 0:
        st.sidebar.warning("⚠️ まず使用するカラム（P1.. / D1..）を選択してください。")
        st.stop() 
    
    init_type = st.sidebar.radio("重みの初期値を選択", ["全て1", "ランダム"], index=0)

    if init_type == "ランダム":
        min_val = st.sidebar.number_input("ランダム最小値", value=-1.0, step=0.1)
        max_val = st.sidebar.number_input("ランダム最大値", value=1.0, step=0.1)
        if min_val >= max_val:
            st.sidebar.warning("最小値は最大値より小さく設定してください。")
            max_val = min_val + 1.0
        weights_init = np.random.uniform(min_val, max_val, n_features)
        weights_init = np.round(weights_init, 1)  # 小数第1位に丸め
    else:
        weights_init = np.ones(n_features, float)

    # 1行目がstock名、2行目がweights_initのDataFrameを作成
    df_weights_init = pd.DataFrame([weights_init], columns=feature_names)
    df_weights_init.index = ["初期重み"]  # 行名を付けると見やすい

    st.markdown("#### 初期重みデータフレーム")
    st.dataframe(df_weights_init)

    weights_init_payload = np.asarray(weights_init, dtype=float).tolist()

    # 欠損値補完のUI（元のまま）
    st.sidebar.selectbox('欠損値補完の方法は？', ['欠損値削除', '中央値補完', '平均値補完', 'k-NN法補完'], index = None, placeholder="選択してください")

    if st.button("開始", help="実験の実行"):
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

        # 重み適用
        df_nociceptive_train_weighted = df_nociceptive_train.mul(weights, axis=1)
        df_neuronociceptive_train_weighted = df_neuronociceptive_train.mul(weights, axis=1)
        df_unknown_train_weighted = df_unknown_train.mul(weights, axis=1)

        datas = np.vstack(
            [df_nociceptive_train_weighted.values, df_neuronociceptive_train_weighted.values, df_unknown_train_weighted.values]
        ).astype(np.float32)

        labels1 = np.full(len(df_nociceptive_train_weighted), 1, np.int32)
        labels2 = np.full(len(df_neuronociceptive_train_weighted), 2, np.int32)
        labels3 = np.full(len(df_unknown_train_weighted), 3, np.int32)
        labels = np.concatenate([labels1, labels2, labels3]).astype(np.int32)

        st.title("🧠 Hill Climbing × 並列探索（SVM最適化）")

        # ==== グリッドの構築 ====
        if not C_values:
            C_values_local = [1.0]
        else:
            C_values_local = C_values

        gamma_vals = gamma_values if gamma_values else [None if kernel in ["linear"] else 0.1]
        degree_vals = degree_values if degree_values else [None if kernel != "poly" else 3]
        coef0_vals = coef0_values if coef0_values else [None if kernel not in ["poly", "sigmoid"] else 0.0]

        param_grid = [
            (step_size, g, d, c0, C)
            for step_size in step_sizes
            for g in gamma_vals
            for d in degree_vals
            for c0 in coef0_vals
            for C in C_values_local
        ]

        all_results = []
        best_score = -np.inf
        best_result = None

        st.write("🔁 並列実行中...")
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(run_hill, step_size, kernel, g, d, c0, C, datas, labels, weights_init_payload, max_iter_hc=max_iter_hc, k=k_cv, max_iter_svc=max_iter_svc):
                (step_size, g, d, c0, C)
                for (step_size, g, d, c0, C) in param_grid
            }
            total_jobs = len(futures)

            done = 0
            progress_bar = st.progress(0, text="進捗状況 0/0（0％）")
            progress_text = st.empty()

            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)

                if result["score"] > best_score:
                    best_score = result["score"]
                    best_result = result

                done += 1
                percent = int(done / total_jobs * 100) if total_jobs > 0 else 100
                progress_bar.progress(percent, text=f"進捗状況 {done}/{total_jobs}（{percent}％）")
                progress_text.text(f"進捗状況 {done}/{total_jobs}")

        elapsed = time.time() - start_time
        st.write(f"⏱ 実行時間: {elapsed:.2f} 秒")

        st.subheader("📊 スコアまとめ（降順）")
        results_df = pd.DataFrame([{
            "step_size": r["step_size"],
            "kernel": r["kernel"],
            "gamma": r["gamma"],
            "degree": r["degree"],
            "coef0": r["coef0"],
            "C": r["C"],
            "score": r["score"],
            "weights": r["weights"]
        } for r in all_results])
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

        # モデル保存（選択カーネルで学習）
        if best_result:
            X_weighted_final = apply_weights(datas, np.array(best_result["weights"]))
            final_model = build_svc(
                kernel=best_result["kernel"],
                C=best_result["C"],
                gamma=best_result["gamma"],
                degree=best_result["degree"],
                coef0=best_result["coef0"],
                max_iter=max_iter_svc
            )
            final_model.fit(X_weighted_final, labels)
            joblib.dump(final_model, "final_model.joblib")
            st.success("✅ 最終モデルを保存しました！（final_model.joblib）")

            best_weights_df = pd.DataFrame(np.array(best_result["weights"]).astype(float), index=feature_names, columns=["Weight"])

            st.subheader("📊 パラメータ別スコア（簡易表）")
            small_df = results_df[["step_size", "kernel", "C", "gamma", "degree", "coef0", "score"]].copy()
            small_df["score"] = (small_df["score"] * 100).map(lambda x: f"{x:.2f}%")
            st.dataframe(small_df.sort_values(by="score", ascending=False))

            st.write("✅ 最適なkernel:", best_result["kernel"])
            st.write("✅ 最適なC:", best_result["C"])
            if best_result["gamma"] is not None:
                st.write("✅ 最適なgamma:", best_result["gamma"])
            if best_result["degree"] is not None:
                st.write("✅ 最適なdegree:", best_result["degree"])
            if best_result["coef0"] is not None:
                st.write("✅ 最適なcoef0:", best_result["coef0"])
            st.write("✅ 最適な重み:")
            st.dataframe(best_weights_df)
            st.write("✅ 最終スコア:", best_score)

            best_y_val = best_result["y_val"]
            best_pred = best_result["pred"]

            conf_matrix = confusion_matrix(best_y_val, best_pred, labels=[1, 2, 3])

            n_classes = conf_matrix.shape[0]
            for i in range(n_classes):
                TP = conf_matrix[i, i]
                FN = np.sum(conf_matrix[i, :]) - TP
                FP = np.sum(conf_matrix[:, i]) - TP
                TN = np.sum(conf_matrix) - (TP + FN + FP)

                sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
                specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

                st.write(f"疼痛 {i+1}: 感度 = {sensitivity * 100:.2f}%, 特異度 = {specificity * 100:.2f}%")

