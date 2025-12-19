import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. 共通ユーティリティ & 定数
# ==========================================

def parse_num_list(s, dtype=float):
    if not s: return []
    out = []
    for chunk in s.replace("，", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            try:
                out.append(dtype(chunk))
            except:
                pass
    return out

def apply_weights(datas, weights):
    return datas * weights

def vec_to_params(vec, kernel):
    p = {"kernel": kernel}
    idx = 0
    p["C"] = vec[idx]; idx += 1
    if kernel in ["rbf", "poly", "sigmoid"]:
        if idx < len(vec): p["gamma"] = vec[idx]; idx += 1
    if kernel in ["poly", "sigmoid"]:
        if idx < len(vec): p["coef0"] = vec[idx]; idx += 1
    return p

def normalize_w(w, clip_low=0.0, clip_high=3.0, mean_one=True):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, clip_low, clip_high)
    if mean_one:
        m = w.mean()
        if m > 0:
            w = w / m
    return w

# ==========================================
# 2. ランダムジャンプロジック
# ==========================================

# 重み用: 部分ジャンプ
def partial_random_jump_w(w, k_pct, strength, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    w_new = np.asarray(w, dtype=float).copy()
    d = len(w_new)
    
    k = max(1, int(np.ceil((k_pct / 100.0) * d)))
    k = min(k, d)
    
    sel = rng.choice(np.arange(d), size=k, replace=False)
    noise = rng.uniform(strength[0], strength[1], size=k)
    
    w_new[sel] *= noise
    return normalize_w(w_new)

# ★★★ パラメータ用: 全て線形スケールでジャンプ ★★★
def random_jump_params_bounded(params_vec, param_types, bounds_list, jump_pct, rng=None):
    """
    Args:
        jump_pct: 範囲に対して何％動かすか (例: 0.2 なら 範囲の20%分をプラスマイナス)
    """
    rng = np.random.default_rng() if rng is None else rng
    p = np.asarray(params_vec, dtype=float).copy()
    d = len(p)
    
    for idx in range(d):
        min_v, max_v = bounds_list[idx]
        
        # 全て Linear Scale として処理
        linear_range = max_v - min_v
        
        # 範囲の ±pct 分だけずらす (一様分布)
        delta = rng.uniform(-jump_pct, jump_pct) * linear_range
        new_val = p[idx] + delta
        
        # 範囲内にクリップ
        p[idx] = np.clip(new_val, min_v, max_v)
                
    return p

# ==========================================
# 3. 評価関数
# ==========================================
def evaluate_core(weights, datas, labels, kernel, params_vec, degree, k=5, max_iter_svc=5000):
    X_weighted = apply_weights(datas, weights)
    p_dict = vec_to_params(params_vec, kernel)
    p_dict["degree"] = degree
    p_dict["max_iter"] = max_iter_svc

    # ★追加: SVCの乱数を固定して再現性を担保する
    p_dict["random_state"] = 42
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in skf.split(X_weighted, labels):
        X_train, X_val = X_weighted[train_index], X_weighted[val_index]
        y_train, y_val = labels[train_index], labels[val_index]
        model = SVC(**p_dict)
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))
    return np.mean(scores)

# ==========================================
# 4. 最適化エンジン (山登り法)
# ==========================================

# --- A. 重みの最適化 ---
def optimize_weights_hc(datas, labels, kernel, current_params_vec, degree, init_weights, step_size, max_iter, k_cv, stagnate_L, jump_settings):
    n_features = len(init_weights)
    
    # 現在地(curr)と、最高記録(global_best)を分離して初期化
    curr_weights = normalize_w(init_weights.copy())
    curr_score = evaluate_core(curr_weights, datas, labels, kernel, current_params_vec, degree, k=k_cv)
    
    global_best_weights = curr_weights.copy()
    global_best_score = curr_score
    
    no_improve = 0
    
    k_small, k_big = jump_settings["k_small"], jump_settings["k_big"]
    str_small, str_big = jump_settings["str_small"], jump_settings["str_big"]
    
    for _ in range(max_iter):
        step_best_score = -np.inf
        step_best_weights = None
        
        # --- 近傍探索 ---
        for idx in range(n_features):
            for delta in [-step_size, 0.0, step_size]:
                if delta == 0.0:
                    score = curr_score
                    trial_w = curr_weights
                else:
                    trial_w = curr_weights.copy()
                    trial_w[idx] += delta
                    trial_w = normalize_w(trial_w)
                    score = evaluate_core(trial_w, datas, labels, kernel, current_params_vec, degree, k=k_cv)
                
                if score > step_best_score:
                    step_best_score = score
                    step_best_weights = trial_w
        
        # --- 移動判定 ---
        if step_best_score > curr_score:
            # 改善した場合、現在地を更新
            curr_score = step_best_score
            curr_weights = step_best_weights
            no_improve = 0
            
            # もし今回の移動で「史上最高記録」を更新したら、Global Bestに保存
            if curr_score > global_best_score:
                global_best_score = curr_score
                global_best_weights = curr_weights.copy()
        else:
            no_improve += 1
            
        # --- ジャンプ処理 ---
        if no_improve >= stagnate_L:
            if no_improve >= stagnate_L * 2:
                # 停滞時、curr(現在地)は強制的に飛ばすが、global_bestは守られる
                curr_weights = partial_random_jump_w(curr_weights, k_big, str_big)
                no_improve = 0
            else:
                curr_weights = partial_random_jump_w(curr_weights, k_small, str_small)
            
            # ジャンプ先のスコアを計算 (これは低くなっても良い)
            curr_score = evaluate_core(curr_weights, datas, labels, kernel, current_params_vec, degree, k=k_cv)

    # 【最終的に、探索中に見つけた一番良いものを返す
    return global_best_weights, global_best_score

# --- B. パラメータの最適化 (All Linear) ---
def optimize_params_hc(datas, labels, kernel, degree, fixed_weights, init_params_vec, step_sizes_vec, max_iter, k_cv, stagnate_L, jump_settings, param_bounds_list):
    
    # 現在地(curr)と、最高記録(global_best)を分離して初期化
    curr_params = np.array(init_params_vec, dtype=float)
    curr_score = evaluate_core(fixed_weights, datas, labels, kernel, curr_params, degree, k=k_cv)
    
    global_best_params = curr_params.copy()
    global_best_score = curr_score
    
    n_params = len(curr_params)
    param_types = ['linear'] * n_params
        
    no_improve = 0
    pct_small, pct_big = jump_settings["pct_small"], jump_settings["pct_big"]
    
    for _ in range(max_iter):
        step_best_score = -np.inf
        step_best_params = None
        
        # --- 近傍探索 ---
        for idx in range(n_params):
            current_val = curr_params[idx]
            step = step_sizes_vec[idx]
            
            cands = [current_val - step, current_val, current_val + step]
            
            for val in cands:
                if val == current_val:
                    score = curr_score
                    trial_p = curr_params
                else:
                    # ガード処理
                    min_v, max_v = param_bounds_list[idx]
                    val = max(val, min_v)
                    val = min(val, max_v)
                    
                    trial_p = curr_params.copy()
                    trial_p[idx] = val
                    score = evaluate_core(fixed_weights, datas, labels, kernel, trial_p, degree, k=k_cv)
                
                if score > step_best_score:
                    step_best_score = score
                    step_best_params = trial_p
        
        # --- 移動判定 ---
        if step_best_score > curr_score:
            curr_score = step_best_score
            curr_params = step_best_params
            no_improve = 0
            
            # 史上最高記録を更新したら保存
            if curr_score > global_best_score:
                global_best_score = curr_score
                global_best_params = curr_params.copy()
        else:
            no_improve += 1
            
        # --- ジャンプ処理 ---
        if no_improve >= stagnate_L:
            if no_improve >= stagnate_L * 2:
                # 現在地のみジャンプ
                curr_params = random_jump_params_bounded(curr_params, param_types, param_bounds_list, pct_big)
                no_improve = 0
            else:
                curr_params = random_jump_params_bounded(curr_params, param_types, param_bounds_list, pct_small)
            
            curr_score = evaluate_core(fixed_weights, datas, labels, kernel, curr_params, degree, k=k_cv)
            
    # 探索中に見つけた一番良いものを返す
    return global_best_params, global_best_score

# --- 並列実行用ラッパー ---
def run_params_search_wrapper(datas, labels, kernel, degree, fixed_weights, init_vec, step_vec, max_iter, k_cv, stagnate_L, jump_settings, bounds_list):
    return optimize_params_hc(datas, labels, kernel, degree, fixed_weights, init_vec, step_vec, max_iter, k_cv, stagnate_L, jump_settings, bounds_list)

def run_weights_search_wrapper(datas, labels, kernel, params_vec, degree, init_weights, step_size, max_iter, k_cv, stagnate_L, jump_settings):
    return optimize_weights_hc(datas, labels, kernel, params_vec, degree, init_weights, step_size, max_iter, k_cv, stagnate_L, jump_settings)

# ==========================================
# 5. メインUI & 実行ロジック
# ==========================================

def main():
    st.set_page_config(page_title="SVM交互最適化", layout="wide")
    st.sidebar.header("SVMパラメータ最適化設定")
    kernel = st.sidebar.selectbox("SVMカーネル", ["linear", "rbf", "poly", "sigmoid"], index=1)

    # --- 初期値候補 ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**初期値候補（並列用）**")
    c_candidates_str = st.sidebar.text_input("C 初期値候補", "0.1, 1, 10.0")
    gamma_candidates_str = "0.01, 0.1, 0.3"
    coef0_candidates_str = "0.0, 1.0, 5.0"
    if kernel in ["rbf", "poly", "sigmoid"]:
        gamma_candidates_str = st.sidebar.text_input("Gamma 初期値候補", "0.01, 0.1, 0.3")
    if kernel in ["poly", "sigmoid"]:
        coef0_candidates_str = st.sidebar.text_input("Coef0 初期値候補", "0.0, 1.0, 5.0")

    candidates_C = parse_num_list(c_candidates_str)
    candidates_gamma = parse_num_list(gamma_candidates_str)
    candidates_coef0 = parse_num_list(coef0_candidates_str)
    if not candidates_C: candidates_C = [1.0]
    if not candidates_gamma: candidates_gamma = [0.1]
    if not candidates_coef0: candidates_coef0 = [0.0]

    # --- ステップ幅 (Linear用に表記とデフォルト値を変更) ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**山登り法の基本設定**")
    
    step_C = st.sidebar.number_input("Step Size (C) ※加算", value=1.0, step=0.1, help="現在値 ± この値 で探索します")
    step_gamma = 0.01
    step_coef0 = 0.1
    
    if kernel in ["rbf", "poly", "sigmoid"]:
        # Gammaは値が小さいことが多いのでデフォルト小さめに
        step_gamma = st.sidebar.number_input("Step Size (gamma) ※加算", value=0.05, step=0.01, format="%.3f")
    if kernel in ["poly", "sigmoid"]:
        step_coef0 = st.sidebar.number_input("Step Size (coef0) ※加算", value=0.5, step=0.05)

    step_weight = st.sidebar.number_input("重み探索ステップ幅 (±val)", 0.01, 0.5, 0.05)
    max_iter_hc = st.sidebar.number_input("反復回数", 10, 5000, 200, 50)
    stagnate_L = st.sidebar.number_input("停滞判定ステップ L", 2, 200, 10)
    k_cv = st.sidebar.slider("CV分割数 (k)", 2, 10, 5)

    # --- ランダムジャンプ詳細設定 ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("ランダムジャンプ設定", expanded=True):
        
        st.markdown("### 1. 特徴量の重み (Weights)")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            k_small_pct = st.number_input("小ジャンプ数(%)", 1, 50, 10)
            w_str_small = st.slider("W強度(小)", 0.5, 1.5, (0.85, 1.15), 0.01)
        with col_w2:
            k_big_pct = st.number_input("大ジャンプ数(%)", 1, 50, 20)
            w_str_big = st.slider("W強度(大)", 0.3, 2.0, (0.5, 1.5), 0.01)

        st.markdown("---")
        st.markdown("### 2. SVMパラメータ")
        st.caption("全パラメータ共通: 範囲ベースの線形ジャンプ")
        
        # 範囲設定 (Bounds)
        st.markdown("**パラメータ有効範囲**")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            c_min = st.number_input("C Min", value=0.01, format="%.2f")
            g_min = st.number_input("Gamma Min", value=0.01, format="%.2f")
            cf_min = st.number_input("Coef0 Min", value=0.0, format="%.2f")
        with col_b2:
            c_max = st.number_input("C Max", value=100.0, format="%.1f")
            g_max = st.number_input("Gamma Max", value=1.0, format="%.1f")
            cf_max = st.number_input("Coef0 Max", value=10.0, format="%.1f")
            
        bounds_dict = {
            "C": (c_min, c_max),
            "gamma": (g_min, g_max),
            "coef0": (cf_min, cf_max)
        }

        # ジャンプ率設定 (%)
        st.markdown("**強度**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p_pct_small = st.number_input("小ジャンプ(%)", 1.0, 50.0, 10.0, step=1.0) / 100.0
        with col_p2:
            p_pct_big = st.number_input("大ジャンプ(%)", 1.0, 100.0, 20.0, step=5.0) / 100.0

        jump_settings_w = {"k_small": k_small_pct, "k_big": k_big_pct, "str_small": w_str_small, "str_big": w_str_big}
        jump_settings_p = {"pct_small": p_pct_small, "pct_big": p_pct_big}


    st.sidebar.markdown("---")
    n_components = st.sidebar.slider("PCA主成分数", 2, 20, 5)
    n_workers = st.sidebar.slider("並列処理コア数", 1, 8, 4)
    st.sidebar.markdown("---")
    n_cycles = st.sidebar.number_input("サイクル回数", 1, 20, 3)

    # --- データ読み込み ---
    st.sidebar.header("データセット設定")
    options_miss = ['欠損値データ削除', '中央値補完', '平均値補完', 'k-NN法補完']
    choice_1 = st.sidebar.selectbox('欠損値の対応', options_miss, index=None)
    options_sheet = ['PainDITECT', 'BS-POP', 'FUSION']
    choice_2 = st.sidebar.selectbox('使用する質問表', options_sheet, index=None)

    if choice_1 is None or choice_2 is None:
        st.info("データセットを選択してください")
        st.stop()
    
    try:
        if choice_1 == '欠損値データ削除' and choice_2 == 'PainDITECT':
            df1 = pd.read_csv('data/null/peindetect/questionnaire_paindetect_missing.csv', encoding='utf-8')
            X_cols = df1.loc[:, "P1":"P13"].columns.tolist(); pain_col = df1.columns[1]
        elif choice_1 == '欠損値データ削除' and choice_2 == 'BS-POP':
            df1 = pd.read_csv('data/null/BSPOP/questionnaire_bspop_missing.csv', encoding='utf-8')
            X_cols = df1.loc[:, "D1":"D18"].columns.tolist(); pain_col = df1.columns[1]
        elif choice_1 == '欠損値データ削除' and choice_2 == 'FUSION':
            df1 = pd.read_csv('data/null/fusion/questionnaire_fusion_missing.csv', encoding='utf-8')
            X_cols = df1.loc[:, "P1":"D18"].columns.tolist(); pain_col = df1.columns[1]
        elif choice_1 == '中央値補完' and choice_2 == 'PainDITECT':
            df1 = pd.read_csv('data/主成分分析用/questionnaire_paindetect_median.csv', encoding='utf-8')
            X_cols = df1.loc[:, "P1":"D13"].columns.tolist(); pain_col = df1.columns[1]
        elif choice_1 == '中央値補完' and choice_2 == 'BS-POP':
            df1 = pd.read_csv('data/主成分分析用/questionnaire_bspop_median.csv', encoding='utf-8')
            X_cols = df1.loc[:, "D1":"D18"].columns.tolist(); pain_col = df1.columns[1]
        elif choice_1 == '中央値補完' and choice_2 == 'FUSION':
            df1 = pd.read_csv('data/主成分分析用/questionnaire_fusion_median.csv', encoding='utf-8')
            X_cols = df1.loc[:, "P1":"D18"].columns.tolist(); pain_col = df1.columns[1]
        elif choice_1 == '平均値補完' and choice_2 == 'PainDITECT':
            df1 = pd.read_csv('data/主成分分析用/questionnaire_paindetect_mean.csv', encoding='utf-8')
            X_cols = df1.loc[:, "P1":"D13"].columns.tolist(); pain_col = df1.columns[1]
        elif choice_1 == '平均値補完' and choice_2 == 'BS-POP':
            df1 = pd.read_csv('data/主成分分析用/questionnaire_bspop_mean.csv', encoding='utf-8')
            X_cols = df1.loc[:, "D1":"D18"].columns.tolist(); pain_col = df1.columns[1]
        elif choice_1 == '平均値補完' and choice_2 == 'FUSION':
            df1 = pd.read_csv('data/主成分分析用/questionnaire_fusion_mean.csv', encoding='utf-8')
            X_cols = df1.loc[:, "P1":"D18"].columns.tolist(); pain_col = df1.columns[1]
        elif choice_1 == 'k-NN法補完' and choice_2 == 'PainDITECT':
            df1 = pd.read_csv('data/主成分分析用/questionnaire_paindetect_knn.csv', encoding='utf-8')
            X_cols = df1.loc[:, "P1":"D13"].columns.tolist(); pain_col = df1.columns[1]
        elif choice_1 == 'k-NN法補完' and choice_2 == 'BS-POP':
            df1 = pd.read_csv('data/主成分分析用/questionnaire_bspop_knn.csv', encoding='utf-8')
            X_cols = df1.loc[:, "D1":"D18"].columns.tolist(); pain_col = df1.columns[1]
        elif choice_1 == 'k-NN法補完' and choice_2 == 'FUSION':
            df1 = pd.read_csv('data/主成分分析用/questionnaire_fusion_knn.csv', encoding='utf-8')
            X_cols = df1.loc[:, "P1":"D18"].columns.tolist(); pain_col = df1.columns[1]
        else:
            st.warning("データセットを選択してください。")
            st.stop()
            
        X = df1[X_cols].copy()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # --- 前処理 ---
    options_std = ['する', 'しない']
    choice_4 = st.sidebar.selectbox('データの標準化', options_std, index=0)
    if choice_4 == "する":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    pca = PCA(n_components, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)
    pca_cols = [f"PCA{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df1.index)
    df_pca_final = pd.concat([df1[[pain_col]], df_pca], axis=1)
    feature_names = pca_cols
    st.success("PCA 実行完了")

    st.sidebar.markdown("### 特徴量の重み (初期値)")
    if "weights" not in st.session_state:
        st.session_state.weights = {col: 1.0 for col in feature_names}
    if st.button("重みをリセット"):
        for col in feature_names:
            st.session_state.weights[col] = 1.0
    weights_list = []
    for col in feature_names:
        val = st.sidebar.slider(f"{col} weight", 0.0, 3.0, st.session_state.weights.get(col, 1.0), 0.01, key=f"w_{col}")
        st.session_state.weights[col] = val
        weights_list.append(val)
    initial_weights = np.array(weights_list)

    # ==========================================
    # 実行ロジック
    # ==========================================
    if st.button("🔄 交互最適化を開始"):
        start_time = time.time()
        
        df_nociceptive = df_pca_final[df_pca_final[pain_col] == "侵害受容性疼痛"]
        df_neuropathic = df_pca_final[df_pca_final[pain_col] == "神経障害性疼痛"]
        df_other = df_pca_final[~df_pca_final[pain_col].isin(["侵害受容性疼痛", "神経障害性疼痛"])]
        
        X1 = df_nociceptive[feature_names].values
        X2 = df_neuropathic[feature_names].values
        X3 = df_other[feature_names].values
        datas = np.vstack([X1, X2, X3]).astype(np.float32)
        l1 = np.full(len(X1), 1, dtype=int); l2 = np.full(len(X2), 2, dtype=int); l3 = np.full(len(X3), 3, dtype=int)
        labels = np.concatenate([l1, l2, l3])

        st.title("SVM & Weights 交互最適化")
        st.info(f"並列処理数: {n_workers}")
        
        history_log = []
        current_weights = normalize_w(initial_weights)
        current_best_params_vec = None
        current_best_degree = 3
        
        step_vec = [step_C]
        if kernel != "linear": step_vec.append(step_gamma)
        if kernel in ["poly", "sigmoid"]: step_vec.append(step_coef0)
        
        # Bounds
        bounds_list = [bounds_dict["C"]]
        if kernel in ["rbf", "poly", "sigmoid"]: bounds_list.append(bounds_dict["gamma"])
        if kernel in ["poly", "sigmoid"]: bounds_list.append(bounds_dict["coef0"])
        
        total_steps = n_cycles * 2
        prog_bar = st.progress(0)
        status_text = st.empty()
        
        for cycle in range(1, n_cycles + 1):
            
            # --- Phase 1: SVM Params ---
            status_text.write(f"Cycle {cycle}: SVMパラメータ最適化中...")
            futures_input = []
            
            if cycle == 1 or current_best_params_vec is None:
                deg_list = [2, 3] if kernel == "poly" else [3]
                import itertools
                cycle_deg = itertools.cycle(deg_list)
                for i in range(n_workers):
                    d = next(cycle_deg)
                    init_vec = []
                    # C
                    if i == 0: val_c = candidates_C[0]
                    else: val_c = random.choice(candidates_C)
                    init_vec.append(val_c)
                    # Gamma
                    if kernel in ["rbf", "poly", "sigmoid"]:
                        if i == 0: val_g = candidates_gamma[0]
                        else: val_g = random.choice(candidates_gamma)
                        init_vec.append(val_g)
                    # Coef0
                    if kernel in ["poly", "sigmoid"]:
                        if i == 0: val_cf = candidates_coef0[0]
                        else: val_cf = random.choice(candidates_coef0)
                        init_vec.append(val_cf)
                    futures_input.append({"degree": d, "init": init_vec})
            else:
                for i in range(n_workers):
                    d = current_best_degree
                    p = current_best_params_vec.copy()
                    if i > 0:
                        # 全てLinear扱いとしてジャンプ
                        p_types = ['linear'] * len(p)
                        p = random_jump_params_bounded(p, p_types, bounds_list, jump_settings_p["pct_small"])
                    futures_input.append({"degree": d, "init": p})

            results = Parallel(n_jobs=len(futures_input))(
                delayed(run_params_search_wrapper)(
                    datas, labels, kernel, inp["degree"], current_weights, inp["init"], 
                    step_vec, max_iter_hc, k_cv, stagnate_L, jump_settings_p, bounds_list
                ) for inp in futures_input
            )
            
            cycle_best_score_p = -np.inf
            cycle_best_res_p = None
            for i, (p_vec, score) in enumerate(results):
                if score > cycle_best_score_p:
                    cycle_best_score_p = score
                    cycle_best_res_p = (p_vec, futures_input[i]["degree"])
            
            current_best_params_vec = cycle_best_res_p[0]
            current_best_degree = cycle_best_res_p[1]
            
            history_log.append({
                "cycle": cycle, "phase": "Params", "score": cycle_best_score_p,
                "weights": current_weights.copy(),
                "params": vec_to_params(current_best_params_vec, kernel)
            })
            prog_bar.progress(((cycle - 1) * 2 + 1) / total_steps)
            
            # --- Phase 2: Weights ---
            status_text.write(f"Cycle {cycle}: 特徴量重み最適化中...")
            futures_w = []
            for i in range(n_workers):
                if i == 0:
                    w_init = current_weights.copy()
                else:
                    w_init = partial_random_jump_w(current_weights, k_pct=20, strength=(0.9, 1.1))
                futures_w.append(w_init)
            
            results_w = Parallel(n_jobs=n_workers)(
                delayed(run_weights_search_wrapper)(
                    datas, labels, kernel, current_best_params_vec, current_best_degree,
                    w_init, step_weight, max_iter_hc, k_cv, stagnate_L, jump_settings_w
                ) for w_init in futures_w
            )
            
            best_score_w = -np.inf
            best_w_final = None
            for w, s in results_w:
                if s > best_score_w:
                    best_score_w = s
                    best_w_final = w
            
            current_weights = best_w_final
            
            history_log.append({
                "cycle": cycle, "phase": "Weights", "score": best_score_w,
                "weights": current_weights.copy(),
                "params": vec_to_params(current_best_params_vec, kernel)
            })
            prog_bar.progress(((cycle - 1) * 2 + 2) / total_steps)
            
            st.write(f"✅ Cycle {cycle} | Params: {cycle_best_score_p:.4f} -> Weights: {best_score_w:.4f}")

        st.success("最適化完了！")
        final = history_log[-1]
        st.subheader(f"🏆 最終スコア: {final['score']:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 最終パラメータ")
            final_p = final["params"]
            final_p["degree"] = current_best_degree
            st.json(final_p)
        with col2:
            st.markdown("#### 最終重み (Top 10)")
            w_df = pd.DataFrame(final["weights"], index=feature_names, columns=["Weight"])
            st.dataframe(w_df)
            
        scores = [h["score"] for h in history_log]
        labels_x = [f"C{h['cycle']}-{h['phase'][0]}" for h in history_log]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(scores, marker="o")
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(labels_x)
        ax.set_title("Optimization History")
        ax.grid(True)
        st.pyplot(fig)
        
        final_X = apply_weights(datas, final["weights"])
        model_p = final["params"]
        model_p["degree"] = current_best_degree
        model_p["max_iter"] = 2000
        final_model = SVC(**model_p)
        final_model.fit(final_X, labels)
        joblib.dump(final_model, "cyclic_optimized_model.joblib")
        st.info("モデル保存完了: cyclic_optimized_model.joblib")

        # 最適化されたパラメータでCVを行う
        cv_model = SVC(**model_p)
        kf_final = StratifiedKFold(n_splits=k_cv, shuffle=True, random_state=42)
        
        # 全データに対する予測値を取得
        best_pred = cross_val_predict(cv_model, final_X, labels, cv=kf_final)
        best_y_val = labels
        
        cm = confusion_matrix(best_y_val, best_pred, labels=[1, 2, 3])
        n_classes = cm.shape[0]
        
        for i in range(n_classes):
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            FP = np.sum(cm[:, i]) - TP
            TN = np.sum(cm) - (TP + FN + FP)

            sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

            st.write(f"疼痛 {i+1}: 感度 = {sensitivity * 100:.2f}%, 特異度 = {specificity * 100:.2f}%") 

if __name__ == "__main__":
    main()