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
# PCAのインポートを削除

# =========================
# ユーティリティ
# =========================

def normalize_w(w, clip_low=0.0, clip_high=3.0, mean_one=True):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, clip_low, clip_high)
    if mean_one:
        m = w.mean()
        if m > 0:
            w = w / m
    return w

def partial_random_jump_w(w, k, strength, *, clip_low=0.0, clip_high=3.0, mean_one=True, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    w = np.asarray(w, dtype=float).copy()
    d = len(w)
    k = max(1, min(int(np.ceil(k)), d))
    sel = rng.choice(np.arange(d), size=k, replace=False)
    noise = rng.uniform(strength[0], strength[1], size=k)
    w[sel] *= noise
    return normalize_w(w, clip_low=clip_low, clip_high=clip_high, mean_one=mean_one)

def parse_num_list(s, dtype=float):
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

# 山登り法(2方向/3方向) は変更なしのため中身は維持

def hill_climbing_1(datas, labels, C, kernel, gamma, degree, coef0, weights_init, max_iter_1=1000, step_size=0.01, k=5, max_iter_svc=1500, stagnate_L=20, k_small_pct=10, k_big_pct=20, small_strength=(0.85, 1.15), big_strength=(0.5, 1.5), mean_one=True):
    CLIP_LOW, CLIP_HIGH = 0.0, 3.0
    EPS_ACCEPT = 0.0
    rng = np.random.default_rng(0)
    n_features = datas.shape[1]
    if weights_init is None:
        weights_init = np.ones(n_features, dtype=float)
    elif isinstance(weights_init, list):
        weights_init = np.asarray(weights_init, dtype=float)
    weights_change = normalize_w(weights_init, CLIP_LOW, CLIP_HIGH, mean_one)
    best_score, best_X_val, best_y_val, best_pred = evaluate(weights_change, datas, labels, C, kernel, gamma, degree, coef0, k=k, return_best_split=True, max_iter_svc=max_iter_svc)
    best_weights = weights_change.copy()
    score_history = [best_score]
    global_best_score = best_score
    global_best_weights = best_weights.copy()
    global_best_pack = (best_X_val, best_y_val, best_pred)
    no_improve = 0

    for _ in range(max_iter_1):
        step_best_score = -np.inf
        candidates = []
        trial_configs = []
        for idx in range(n_features):
            for delta in [-step_size, step_size]:
                trial_weights = best_weights.copy().astype(float)
                trial_weights[idx] += delta
                trial_weights = normalize_w(trial_weights, CLIP_LOW, CLIP_HIGH, mean_one)
                trial_configs.append(trial_weights)
        results = []
        for trial_weights in trial_configs:
            results.append(evaluate(trial_weights, datas, labels, C, kernel, gamma, degree, coef0, k=k, return_best_split=True, max_iter_svc=max_iter_svc))
        for i in range(len(trial_configs)):
            trial_weights = trial_configs[i]
            score, X_val_tmp, y_val_tmp, pred_tmp = results[i]
            if score > step_best_score:
                step_best_score = score
                candidates = [(trial_weights.copy(), X_val_tmp, y_val_tmp, pred_tmp)]
            elif score == step_best_score:
                candidates.append((trial_weights.copy(), X_val_tmp, y_val_tmp, pred_tmp))
        sel_w, sel_Xv, sel_yv, sel_pred = random.choice(candidates)
        if (step_best_score >= best_score) or (rng.random() < EPS_ACCEPT):
            best_weights = sel_w
            best_score = step_best_score
            best_X_val, best_y_val, best_pred = sel_Xv, sel_yv, sel_pred
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_weights = best_weights.copy()
            global_best_pack = (best_X_val, best_y_val, best_pred)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= stagnate_L:
            d = len(best_weights)
            k_small = max(1, int(np.ceil((k_small_pct/100.0) * d)))
            best_weights = partial_random_jump_w(best_weights, k=k_small, strength=small_strength, clip_low=CLIP_LOW, clip_high=CLIP_HIGH, mean_one=mean_one, rng=rng)
            if no_improve >= 2 * stagnate_L:
                k_big = max(1, int(np.ceil((k_big_pct/100.0) * d)))
                best_weights = partial_random_jump_w(best_weights, k=k_big, strength=big_strength, clip_low=CLIP_LOW, clip_high=CLIP_HIGH, mean_one=mean_one, rng=rng)
                no_improve = 0
        score_history.append(best_score)
    return global_best_weights, global_best_score, global_best_pack[0], global_best_pack[1], global_best_pack[2], score_history

def hill_climbing_2(datas, labels, C, kernel, gamma, degree, coef0, weights_init, max_iter_1=1000, step_size=0.01, k=5, max_iter_svc=1500, stagnate_L=20, k_small_pct=10, k_big_pct=20, small_strength=(0.85, 1.15), big_strength=(0.5, 1.5), mean_one=True):
    CLIP_LOW, CLIP_HIGH = 0.0, 3.0
    EPS_ACCEPT = 0.0
    rng = np.random.default_rng(0)
    n_features = datas.shape[1]
    if weights_init is None:
        weights_init = np.ones(n_features, dtype=float)
    elif isinstance(weights_init, list):
        weights_init = np.asarray(weights_init, dtype=float)
    weights_change = normalize_w(weights_init, CLIP_LOW, CLIP_HIGH, mean_one)
    best_score, best_X_val, best_y_val, best_pred = evaluate(weights_change, datas, labels, C, kernel, gamma, degree, coef0, k=k, return_best_split=True, max_iter_svc=max_iter_svc)
    best_weights = weights_change.copy()
    score_history = [best_score]
    global_best_score = best_score
    global_best_weights = best_weights.copy()
    global_best_pack = (best_X_val, best_y_val, best_pred)
    no_improve = 0
    for i in range(max_iter_1):
        step_best_score = best_score
        step_best_weights = best_weights.copy()
        step_best_pack = (best_X_val, best_y_val, best_pred)
        for idx in range(n_features):
            for delta in (-step_size, 0.0, step_size):
                trial_weights = best_weights.copy()
                trial_weights[idx] += delta
                trial_weights = normalize_w(trial_weights, CLIP_LOW, CLIP_HIGH, mean_one)
                if delta == 0.0:
                    score, Xv, yv, pred = best_score, best_X_val, best_y_val, best_pred
                else:
                    score, Xv, yv, pred = evaluate(trial_weights, datas, labels, C, kernel, gamma, degree, coef0, k=k, return_best_split=True, max_iter_svc=max_iter_svc)
                if score > step_best_score:
                    step_best_score = score
                    step_best_weights = trial_weights
                    step_best_pack = (Xv, yv, pred)
        if (step_best_score > best_score) or (rng.random() < EPS_ACCEPT):
            best_score = step_best_score
            best_weights = step_best_weights
            best_X_val, best_y_val, best_pred = step_best_pack
            if best_score > global_best_score:
                global_best_score = best_score
                global_best_weights = best_weights.copy()
                global_best_pack = step_best_pack
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        if no_improve >= stagnate_L:
            d = len(best_weights)
            k_small = max(1, int(np.ceil((k_small_pct/100.0) * d)))
            best_weights = partial_random_jump_w(best_weights, k=k_small, strength=small_strength, clip_low=CLIP_LOW, clip_high=CLIP_HIGH, mean_one=mean_one, rng=rng)
            if no_improve >= 2 * stagnate_L:
                k_big = max(1, int(np.ceil((k_big_pct/100.0) * d)))
                best_weights = partial_random_jump_w(best_weights, k=k_big, strength=big_strength, clip_low=CLIP_LOW, clip_high=CLIP_HIGH, mean_one=mean_one, rng=rng)
                no_improve = 0
        score_history.append(best_score)
    return global_best_weights, global_best_score, global_best_pack[0], global_best_pack[1], global_best_pack[2], score_history

# ラッパー関数
def run_hill_climbing_1(step_size, kernel, gamma, degree, coef0, C, datas, labels, weights_init, max_iter_hc=1000, k=5, max_iter_svc=1500, stagnate_L=20, k_small_pct=10, k_big_pct=20, small_strength=(0.85, 1.15), big_strength=(0.5, 1.5), mean_one=True):
    weights_best, score, X_val_tmp, y_val_tmp, pred_tmp, score_history = hill_climbing_1(datas, labels, C, kernel, gamma, degree, coef0, weights_init, max_iter_1=max_iter_hc, step_size=step_size, k=k, max_iter_svc=max_iter_svc, stagnate_L=stagnate_L, k_small_pct=k_small_pct, k_big_pct=k_big_pct, small_strength=small_strength, big_strength=big_strength, mean_one=mean_one)
    return {"step_size": step_size, "kernel": kernel, "gamma": gamma, "degree": degree, "coef0": coef0, "C": C, "score": score, "weights": [float(f"{w:.2f}") for w in weights_best], "weights_raw": np.asarray(weights_best, dtype=float).tolist(), "score_history": score_history, "X_val": X_val_tmp, "y_val": y_val_tmp, "pred": pred_tmp}

def run_hill_climbing_2(step_size, kernel, gamma, degree, coef0, C, datas, labels, weights_init, max_iter_hc=1000, k=5, max_iter_svc=1500, stagnate_L=20, k_small_pct=10, k_big_pct=20, small_strength=(0.85, 1.15), big_strength=(0.5, 1.5), mean_one=True):
    weights_best, score, X_val_tmp, y_val_tmp, pred_tmp, score_history = hill_climbing_2(datas, labels, C, kernel, gamma, degree, coef0, weights_init, max_iter_1=max_iter_hc, step_size=step_size, k=k, max_iter_svc=max_iter_svc, stagnate_L=stagnate_L, k_small_pct=k_small_pct, k_big_pct=k_big_pct, small_strength=small_strength, big_strength=big_strength, mean_one=mean_one)
    return {"step_size": step_size, "kernel": kernel, "gamma": gamma, "degree": degree, "coef0": coef0, "C": C, "score": score, "weights": [float(f"{w:.2f}") for w in weights_best], "weights_raw": np.asarray(weights_best, dtype=float).tolist(), "score_history": score_history, "X_val": X_val_tmp, "y_val": y_val_tmp, "pred": pred_tmp}

# =========================
# UI & メイン処理
# =========================
def run_shift_experiment():
    hill_type = st.sidebar.radio("山登り法の選択",["2方向", "3方向"], index=0)

    if hill_type == "2方向":
        run_hill = run_hill_climbing_1
        # 説明UI省略(維持)
    elif hill_type == "3方向":
        run_hill = run_hill_climbing_2
        # 説明UI省略(維持)

    # サイドバー設定
    st.sidebar.header("最適化の設定")
    kernel = st.sidebar.selectbox("SVMカーネル", ["linear", "rbf", "poly", "sigmoid"], index=0)
    step_sizes_str = st.sidebar.text_input("ステップ幅", value="0.01")
    step_sizes = parse_num_list(step_sizes_str, float) or [0.01]
    max_iter_hc = st.sidebar.number_input("山登り法の反復回数", value=1000)
    k_cv = st.sidebar.slider("分割数 (k)", 2, 8, 5)
    max_iter_svc = st.sidebar.number_input("SVC max_iter", value=1500)

    st.sidebar.header("ランダムジャンプの設定")
    stagnate_L = st.sidebar.number_input("停滞判定ステップ L", value=20)
    col1, col2 = st.sidebar.columns(2)
    with col1: k_small_pct = st.number_input("小ジャンプ割合(%)", value=10)
    with col2: k_big_pct = st.number_input("大ジャンプ割合(%)", value=20)
    small_a, small_b = st.slider("a,b(小)", 0.5, 1.5, (0.85, 1.15))
    big_a, big_b = st.slider("a,b(大)", 0.3, 2.0, (0.5, 1.5))
    mean_one = st.sidebar.checkbox("ジャンプ後 平均1に正規化", value=True)

    # PCAコンポーネント数スライダーは削除

    st.sidebar.header("グリッドサーチ値")
    C_values = parse_num_list(st.sidebar.text_input("C", value="0.1, 1, 10"), float)
    gamma_values = parse_num_list(st.sidebar.text_input("gamma", value="0.01, 0.05"), float) if kernel in ["rbf", "poly", "sigmoid"] else []
    degree_values = parse_num_list(st.sidebar.text_input("degree (poly)", value="2, 3"), int) if kernel == "poly" else []
    coef0_values = parse_num_list(st.sidebar.text_input("coef0 (poly/sigmoid)", value="0.0, 0.5"), float) if kernel in ["poly", "sigmoid"] else []

    st.sidebar.header("データセット設定")
    choice_1 = st.sidebar.selectbox('欠損値の対応', ['欠損値データ削除', '中央値補完', '平均値補完', 'k-NN法補完'], index=None)
    choice_2 = st.sidebar.selectbox('使用する質問表', ['PainDITECT', 'BS-POP', 'FUSION'], index=None)

    # データ読み込みロジック (維持)
    X = None
    pain_col = None
    df1 = None

    if choice_1 and choice_2:
        # パス設定の簡略化表示（元のパスを維持）
        if choice_1 == '欠損値データ削除':
            if choice_2 == 'PainDITECT': df1 = pd.read_csv('data/null/peindetect/questionnaire_paindetect_missing.csv')
            elif choice_2 == 'BS-POP': df1 = pd.read_csv('data/null/BSPOP/questionnaire_bspop_missing.csv')
            elif choice_2 == 'FUSION': df1 = pd.read_csv('data/null/fusion/questionnaire_fusion_missing.csv')
        elif choice_1 == '中央値補完':
            if choice_2 == 'PainDITECT': df1 = pd.read_csv('data/主成分分析用/questionnaire_paindetect_median.csv')
            elif choice_2 == 'BS-POP': df1 = pd.read_csv('data/主成分分析用/questionnaire_bspop_median.csv')
            elif choice_2 == 'FUSION': df1 = pd.read_csv('data/主成分分析用/questionnaire_fusion_median.csv')
        elif choice_1 == '平均値補完':
            if choice_2 == 'PainDITECT': df1 = pd.read_csv('data/主成分分析用/questionnaire_paindetect_mean.csv')
            elif choice_2 == 'BS-POP': df1 = pd.read_csv('data/主成分分析用/questionnaire_bspop_mean.csv')
            elif choice_2 == 'FUSION': df1 = pd.read_csv('data/主成分分析用/questionnaire_fusion_mean.csv')
        elif choice_1 == 'k-NN法補完':
            if choice_2 == 'PainDITECT': df1 = pd.read_csv('data/主成分分析用/questionnaire_paindetect_knn.csv')
            elif choice_2 == 'BS-POP': df1 = pd.read_csv('data/主成分分析用/questionnaire_bspop_knn.csv')
            elif choice_2 == 'FUSION': df1 = pd.read_csv('data/主成分分析用/questionnaire_fusion_knn.csv')

        if df1 is not None:
            st.markdown('#### データ')
            st.dataframe(df1)
            pain_col = df1.columns[1]
            if choice_2 == 'PainDITECT': X_cols = df1.loc[:, "P1":"P13" if "P13" in df1.columns else "D13"].columns.tolist()
            elif choice_2 == 'BS-POP': X_cols = df1.loc[:, "D1":"D18"].columns.tolist()
            else: X_cols = df1.loc[:, "P1":"D18"].columns.tolist()
            X = df1[X_cols].copy()

    choice_4 = st.sidebar.selectbox('データの標準化', ['する', 'しない'], index=None)

    if X is not None:
        if choice_4 == "する":
            scaler = StandardScaler()
            X_data = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        else:
            X_data = X.copy()
        
        # --- PCA 関連を削除し、元の X_data をそのまま使用 ---
        feature_names = X_data.columns.tolist()
        df_final = pd.concat([df1[[pain_col]], X_data], axis=1)
        st.success("データの準備が完了しました（PCAなし）")

        # 重み設定 (Session State)
        if "weights" not in st.session_state:
            st.session_state.weights = {col: 1.0 for col in feature_names}
        
        if st.button("重みをリセット"):
            for col in feature_names: st.session_state.weights[col] = 1.0
            st.rerun()

        st.sidebar.markdown("### 重み付け（特徴量別）")
        weights = []
        for col in feature_names:
            w = st.sidebar.slider(f"{col} の重み", -5.0, 5.0, st.session_state.weights.get(col, 1.0), 0.1, key=f"s_{col}")
            st.session_state.weights[col] = w
            weights.append(w)

        edited_df = pd.DataFrame({"columns": feature_names, "weights": weights})
        st.write("現在の重み")
        st.dataframe(edited_df.T)

        # 初期重み設定
        init_type = st.sidebar.radio("重みの初期値を選択", ["全て1", "ランダム"], index=0)
        if init_type == "ランダム":
            min_val = st.sidebar.number_input("ランダム最小値", value=-1.0)
            max_val = st.sidebar.number_input("ランダム最大値", value=1.0)
            weights_init = np.round(np.random.uniform(min_val, max_val, len(feature_names)), 1)
        else:
            weights_init = np.ones(len(feature_names), float)

        if st.button("開始"):
            # 疼痛種類で分割
            df_noc = df_final[df_final[pain_col] == "侵害受容性疼痛"][feature_names]
            df_neu = df_final[df_final[pain_col] == "神経障害性疼痛"][feature_names]
            df_oth = df_final[~df_final[pain_col].isin(["侵害受容性疼痛", "神経障害性疼痛"])][feature_names]

            # スライダーの重みを適用（初期評価用）
            datas = np.vstack([df_noc.mul(weights).values, df_neu.mul(weights).values, df_oth.mul(weights).values]).astype(np.float32)
            labels = np.concatenate([np.full(len(df_noc), 1), np.full(len(df_neu), 2), np.full(len(df_oth), 3)]).astype(np.int32)

            # パラメータグリッド構築
            gamma_vals = gamma_values if gamma_values else [None]
            degree_vals = degree_values if degree_values else [None]
            coef0_vals = coef0_values if coef0_values else [None]
            param_grid = [(ss, g, d, c0, C) for ss in step_sizes for g in gamma_vals for d in degree_vals for c0 in coef0_vals for C in C_values]

            all_results = []
            best_score = -np.inf
            best_result = None

            st.write("🔁 並列実行中...")
            start_time = time.time()
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(run_hill, p[0], kernel, p[1], p[2], p[3], p[4], datas, labels, weights_init.tolist(), max_iter_hc=max_iter_hc, k=k_cv, max_iter_svc=max_iter_svc): p for p in param_grid}
                for future in as_completed(futures):
                    res = future.result()
                    all_results.append(res)
                    if res["score"] > best_score:
                        best_score = res["score"]
                        best_result = res

            st.write(f"⏱ 実行時間: {time.time() - start_time:.2f} 秒")
            
            # 結果表示（維持）
            results_df = pd.DataFrame(all_results)
            st.subheader("📊 スコアまとめ")
            st.dataframe(results_df[["step_size", "kernel", "C", "score"]].sort_values("score", ascending=False))

            if best_result:
                st.subheader("📈 最良の推移")
                fig, ax = plt.subplots()
                ax.plot(best_result["score_history"])
                st.pyplot(fig)
                
                st.write("✅ 最適な重み:")
                st.dataframe(pd.DataFrame(best_result["weights"], index=feature_names, columns=["Weight"]))

                # 感度・特異度算出
                conf_matrix = confusion_matrix(best_result["y_val"], best_result["pred"], labels=[1, 2, 3])
                for i in range(3):
                    TP = conf_matrix[i, i]
                    FN = np.sum(conf_matrix[i, :]) - TP
                    FP = np.sum(conf_matrix[:, i]) - TP
                    TN = np.sum(conf_matrix) - (TP + FN + FP)
                    st.write(f"疼痛 {i+1}: 感度 = {TP/(TP+FN)*100:.2f}%, 特異度 = {TN/(TN+FP)*100:.2f}%")

    else:
        st.info("データセットと標準化の設定を選択してください。")

if __name__ == "__main__":
    run_shift_experiment()