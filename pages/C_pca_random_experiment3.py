import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from joblib import Parallel, delayed
from sklearn.svm import SVC
# ★修正: 正しい場所からインポート
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. 共通ユーティリティ
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
# 2. ランダムジャンプ (線形統一版)
# ==========================================

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

def random_jump_params_linear(params_vec, bounds_list, jump_pct, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    p = np.asarray(params_vec, dtype=float).copy()
    d = len(p)
    
    for idx in range(d):
        min_v, max_v = bounds_list[idx]
        linear_range = max_v - min_v
        
        delta = rng.uniform(-jump_pct, jump_pct) * linear_range
        new_val = p[idx] + delta
        p[idx] = np.clip(new_val, min_v, max_v)
                
    return p

# ==========================================
# 3. 評価関数 (random_state固定)
# ==========================================
def evaluate_core(weights, datas, labels, kernel, params_vec, degree, k=5, max_iter_svc=5000):
    X_weighted = apply_weights(datas, weights)
    p_dict = vec_to_params(params_vec, kernel)
    p_dict["degree"] = degree
    p_dict["max_iter"] = max_iter_svc
    
    # ★重要: SVCの乱数シードを固定
    p_dict["random_state"] = 42
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    # データ型をfloat64に統一
    X_weighted = X_weighted.astype(np.float64)
    
    for train_index, val_index in skf.split(X_weighted, labels):
        X_train, X_val = X_weighted[train_index], X_weighted[val_index]
        y_train, y_val = labels[train_index], labels[val_index]
        
        model = SVC(**p_dict)
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))
        
    return np.mean(scores)

# ==========================================
# 4. 最適化エンジン (並列対応)
# ==========================================

def optimize_weights_hc(datas, labels, kernel, current_params_vec, degree, init_weights, step_size, max_iter, k_cv, stagnate_L, jump_settings):
    n_features = len(init_weights)
    best_weights = normalize_w(init_weights.copy())
    
    best_score = evaluate_core(best_weights, datas, labels, kernel, current_params_vec, degree, k=k_cv)
    no_improve = 0
    
    k_small, k_big = jump_settings["k_small"], jump_settings["k_big"]
    str_small, str_big = jump_settings["str_small"], jump_settings["str_big"]
    
    for _ in range(max_iter):
        step_best_score = -np.inf
        step_best_weights = None
        
        for idx in range(n_features):
            for delta in [-step_size, 0.0, step_size]:
                if delta == 0.0:
                    score = best_score
                    trial_w = best_weights
                else:
                    trial_w = best_weights.copy()
                    trial_w[idx] += delta
                    trial_w = normalize_w(trial_w)
                    score = evaluate_core(trial_w, datas, labels, kernel, current_params_vec, degree, k=k_cv)
                
                if score > step_best_score:
                    step_best_score = score
                    step_best_weights = trial_w
        
        if step_best_score > best_score:
            best_score = step_best_score
            best_weights = step_best_weights
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= stagnate_L:
            if no_improve >= stagnate_L * 2:
                best_weights = partial_random_jump_w(best_weights, k_big, str_big)
                no_improve = 0
            else:
                best_weights = partial_random_jump_w(best_weights, k_small, str_small)
            best_score = evaluate_core(best_weights, datas, labels, kernel, current_params_vec, degree, k=k_cv)

    return best_weights, best_score

def optimize_params_hc(datas, labels, kernel, degree, fixed_weights, init_params_vec, step_sizes_vec, max_iter, k_cv, stagnate_L, jump_settings, param_bounds_list):
    best_params = np.array(init_params_vec, dtype=float)
    best_score = evaluate_core(fixed_weights, datas, labels, kernel, best_params, degree, k=k_cv)
    n_params = len(best_params)
    
    no_improve = 0
    pct_small, pct_big = jump_settings["pct_small"], jump_settings["pct_big"]
    
    for _ in range(max_iter):
        step_best_score = -np.inf
        step_best_params = None
        
        for idx in range(n_params):
            current_val = best_params[idx]
            step = step_sizes_vec[idx]
            cands = [current_val - step, current_val, current_val + step]
            
            for val in cands:
                if val == current_val:
                    score = best_score
                    trial_p = best_params
                else:
                    min_v, max_v = param_bounds_list[idx]
                    val = max(val, min_v)
                    val = min(val, max_v)
                    
                    trial_p = best_params.copy()
                    trial_p[idx] = val
                    score = evaluate_core(fixed_weights, datas, labels, kernel, trial_p, degree, k=k_cv)
                
                if score > step_best_score:
                    step_best_score = score
                    step_best_params = trial_p
        
        if step_best_score > best_score:
            best_score = step_best_score
            best_params = step_best_params
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= stagnate_L:
            if no_improve >= stagnate_L * 2:
                best_params = random_jump_params_linear(best_params, param_bounds_list, pct_big)
                no_improve = 0
            else:
                best_params = random_jump_params_linear(best_params, param_bounds_list, pct_small)
            best_score = evaluate_core(fixed_weights, datas, labels, kernel, best_params, degree, k=k_cv)
            
    return best_params, best_score

def run_params_search_wrapper(datas, labels, kernel, degree, fixed_weights, init_vec, step_vec, max_iter, k_cv, stagnate_L, jump_settings, bounds_list):
    return optimize_params_hc(datas, labels, kernel, degree, fixed_weights, init_vec, step_vec, max_iter, k_cv, stagnate_L, jump_settings, bounds_list)

def run_weights_search_wrapper(datas, labels, kernel, params_vec, degree, init_weights, step_size, max_iter, k_cv, stagnate_L, jump_settings):
    return optimize_weights_hc(datas, labels, kernel, params_vec, degree, init_weights, step_size, max_iter, k_cv, stagnate_L, jump_settings)

# ==========================================
# 5. メインUI
# ==========================================

def main():
    st.set_page_config(page_title="SVM詳細ログ版", layout="wide")
    st.sidebar.header("SVMパラメータ設定")
    kernel = st.sidebar.selectbox("SVMカーネル", ["linear", "rbf", "poly", "sigmoid"], index=1)

    st.sidebar.markdown("**初期値候補 (並列用)**")
    c_cand = parse_num_list(st.sidebar.text_input("C 候補", "0.1, 1, 10.0"))
    g_cand = parse_num_list(st.sidebar.text_input("Gamma 候補", "0.01, 0.1, 0.3")) if kernel in ["rbf", "poly", "sigmoid"] else [0.1]
    cf_cand = parse_num_list(st.sidebar.text_input("Coef0 候補", "0.0, 1.0, 5.0")) if kernel in ["poly", "sigmoid"] else [0.0]
    
    if not c_cand: c_cand = [1.0]
    if not g_cand: g_cand = [0.1]
    if not cf_cand: cf_cand = [0.0]

    st.sidebar.markdown("**山登り法ステップ (線形加算)**")
    step_C = st.sidebar.number_input("Step C", value=1.0, step=0.1)
    step_gamma = st.sidebar.number_input("Step Gamma", value=0.05, step=0.01, format="%.3f")
    step_coef0 = st.sidebar.number_input("Step Coef0", value=0.5, step=0.05)
    step_weight = st.sidebar.number_input("Step Weight", value=0.05, step=0.01)

    max_iter = st.sidebar.number_input("反復回数", 10, 5000, 200)
    stagnate_L = st.sidebar.number_input("停滞判定 L", 2, 200, 10)
    k_cv = st.sidebar.slider("CV分割数", 2, 10, 5)

    with st.sidebar.expander("ジャンプ設定 (範囲ベース)", expanded=True):
        st.markdown("**1. 重み**")
        col1, col2 = st.columns(2)
        k_small = col1.number_input("W小数(%)", 1, 50, 10)
        ws_small = col1.slider("W強度(小)", 0.5, 1.5, (0.85, 1.15), 0.01)
        k_big = col2.number_input("W大数(%)", 1, 50, 20)
        ws_big = col2.slider("W強度(大)", 0.3, 2.0, (0.5, 1.5), 0.01)
        
        st.markdown("**2. パラメータ (範囲有効Min-Max)**")
        col3, col4 = st.columns(2)
        c_min = col3.number_input("C Min", 0.01, 1000.0, 0.01); c_max = col4.number_input("C Max", 0.01, 1000.0, 100.0)
        g_min = col3.number_input("G Min", 0.001, 100.0, 0.001); g_max = col4.number_input("G Max", 0.001, 100.0, 10.0)
        cf_min = col3.number_input("Cf Min", 0.0, 100.0, 0.0); cf_max = col4.number_input("Cf Max", 0.0, 100.0, 10.0)
        
        st.markdown("**ジャンプ率 (範囲の±%)**")
        p_pct_s = st.slider("P小率(%)", 1, 50, 5) / 100.0
        p_pct_b = st.slider("P大率(%)", 1, 100, 20) / 100.0

        j_set_w = {"k_small": k_small, "k_big": k_big, "str_small": ws_small, "str_big": ws_big}
        j_set_p = {"pct_small": p_pct_s, "pct_big": p_pct_b}
        bounds = [(c_min, c_max), (g_min, g_max), (cf_min, cf_max)]

    n_workers = st.sidebar.slider("並列数", 1, 8, 4)
    n_cycles = st.sidebar.number_input("サイクル数", 1, 20, 3)

    st.sidebar.header("データセット")
    opt_miss = ['欠損値データ削除', '中央値補完', '平均値補完', 'k-NN法補完']
    miss = st.sidebar.selectbox('欠損対応', opt_miss, index=None)
    opt_sheet = ['PainDITECT', 'BS-POP', 'FUSION']
    sheet = st.sidebar.selectbox('質問表', opt_sheet, index=None)

    if not miss or not sheet: st.stop()
    
    try:
        if miss == '欠損値データ削除':
             path = f'data/null/{sheet.lower().replace("-","")}/questionnaire_{sheet.lower().replace("-","")}_missing.csv'
        else:
             path = f'data/主成分分析用/questionnaire_{sheet.lower().replace("-","")}_median.csv'
        
        try:
             df1 = pd.read_csv(path, encoding='utf-8')
        except:
             df1 = pd.read_csv('data/主成分分析用/questionnaire_paindetect_median.csv', encoding='utf-8')
        
        if "PainDITECT" in sheet: col_range = slice("P1","P13")
        elif "BS-POP" in sheet: col_range = slice("D1","D18")
        else: col_range = slice("P1","D18")
        
        X = df1.loc[:, col_range].copy()
        pain_col = df1.columns[1]
    except:
        st.error("データ読み込みエラー。ファイルパスを確認してください。")
        st.stop()

    X_scaled = StandardScaler().fit_transform(X)
    n_comp = st.sidebar.slider("PCA数", 2, 20, 5)
    pca = PCA(n_comp)
    X_pca = pca.fit_transform(X_scaled)
    feat_names = [f"PCA{i+1}" for i in range(n_comp)]
    
    df_final = pd.concat([df1[[pain_col]], pd.DataFrame(X_pca, index=df1.index, columns=feat_names)], axis=1)
    
    X1 = df_final[df_final[pain_col] == "侵害受容性疼痛"][feat_names].values
    X2 = df_final[df_final[pain_col] == "神経障害性疼痛"][feat_names].values
    X3 = df_final[~df_final[pain_col].isin(["侵害受容性疼痛", "神経障害性疼痛"])][feat_names].values
    datas = np.vstack([X1, X2, X3]).astype(np.float64)
    labels = np.concatenate([np.full(len(X1),1), np.full(len(X2),2), np.full(len(X3),3)])

    if "weights" not in st.session_state: st.session_state.weights = {c:1.0 for c in feat_names}
    w_init = np.array([st.sidebar.slider(f"{c}", 0.0, 3.0, 1.0, 0.1) for c in feat_names])

    if st.button("開始"):
        st.title("SVM & Weights 整合性チェック付き実行")
        
        curr_w = normalize_w(w_init)
        curr_p = None
        curr_deg = 3
        previous_score = -np.inf
        
        b_list = [bounds[0]] # C
        if kernel != "linear": b_list.append(bounds[1]) # G
        if kernel in ["poly", "sigmoid"]: b_list.append(bounds[2]) # Cf
        
        s_vec = [step_C]
        if kernel != "linear": s_vec.append(step_gamma)
        if kernel in ["poly", "sigmoid"]: s_vec.append(step_coef0)
        
        prog = st.progress(0)
        
        for cy in range(1, n_cycles+1):
            st.markdown(f"## 🌀 Cycle {cy}")
            
            # -----------------------------------------------------
            # Phase 1: SVM Params Optimization
            # -----------------------------------------------------
            st.markdown("### 🔹 Step 1: SVM Params Optimization")
            
            # --- [Step 1 初期状態ログ] ---
            if curr_p is not None:
                # 前回の重み・パラメータでのスコア（初期値）
                init_score_p = evaluate_core(curr_w, datas, labels, kernel, curr_p, curr_deg, k_cv, 5000)
                st.info(f"**Step 1 初期状態 (Start)**: Score = {init_score_p:.6f}")
                
                with st.expander("Step 1 初期値詳細 (Params & Weights)"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("Initial Params:")
                        st.json(vec_to_params(curr_p, kernel))
                    with c2:
                        st.write("Fixed Weights:")
                        st.dataframe(pd.DataFrame(curr_w, index=feat_names, columns=["Weight"]).T)
            else:
                st.info("**Step 1 初期状態**: ランダム初期値からスタート (Score計算なし)")

            # 探索実行
            inits = []
            if cy == 1 or curr_p is None:
                deg_iter = [2,3] if kernel == "poly" else [3]
                import itertools
                cy_deg = itertools.cycle(deg_iter)
                for i in range(n_workers):
                    d = next(cy_deg)
                    vec = [random.choice(c_cand)]
                    if len(b_list)>1: vec.append(random.choice(g_cand))
                    if len(b_list)>2: vec.append(random.choice(cf_cand))
                    inits.append({"d":d, "v":vec})
            else:
                for i in range(n_workers):
                    v = curr_p.copy()
                    if i > 0:
                        v = random_jump_params_linear(v, b_list, j_set_p["pct_small"])
                    inits.append({"d":curr_deg, "v":v})
            
            res = Parallel(n_jobs=n_workers)(
                delayed(run_params_search_wrapper)(
                    datas, labels, kernel, inp["d"], curr_w, inp["v"], s_vec, max_iter, k_cv, stagnate_L, j_set_p, b_list
                ) for inp in inits
            )
            
            best_s = -np.inf
            for i, (pv, sc) in enumerate(res):
                if sc > best_s:
                    best_s = sc
                    curr_p = pv
                    curr_deg = inits[i]["d"]
            
            # --- [Step 1 終了状態ログ] ---
            st.success(f"**Step 1 到達状態 (Max)**: Score = {best_s:.6f}")
            with st.expander("Step 1 最適化結果詳細"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Best Params (Optimized):")
                    st.json(vec_to_params(curr_p, kernel))
                with c2:
                    st.write("Fixed Weights:")
                    st.dataframe(pd.DataFrame(curr_w, index=feat_names, columns=["Weight"]).T)
            
            
            # -----------------------------------------------------
            # Phase 2: Weights Optimization
            # -----------------------------------------------------
            st.markdown("### 🔸 Step 2: Weights Optimization")

            # --- [Step 2 初期状態ログ] ---
            # Step 1のベストパラメータ と 現在の重み(=Step 1と同じ) でのスコア
            # 理論上 best_s と同じになるはず
            init_score_w = evaluate_core(curr_w, datas, labels, kernel, curr_p, curr_deg, k_cv, 5000)
            
            st.info(f"**Step 2 初期状態 (Start)**: Score = {init_score_w:.6f}")
            with st.expander("Step 2 初期値詳細"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Fixed Params:")
                    st.json(vec_to_params(curr_p, kernel))
                with c2:
                    st.write("Initial Weights:")
                    st.dataframe(pd.DataFrame(curr_w, index=feat_names, columns=["Weight"]).T)
            
            w_inits = []
            for i in range(n_workers):
                if i == 0: w_inits.append(curr_w.copy())
                else: w_inits.append(partial_random_jump_w(curr_w, 20, (0.9, 1.1)))
                
            res_w = Parallel(n_jobs=n_workers)(
                delayed(run_weights_search_wrapper)(
                    datas, labels, kernel, curr_p, curr_deg, wi, step_weight, max_iter, k_cv, stagnate_L, j_set_w
                ) for wi in w_inits
            )
            
            best_sw = -np.inf
            for w, s in res_w:
                if s > best_sw:
                    best_sw = s
                    curr_w = w
            
            # --- [Step 2 終了状態ログ] ---
            st.success(f"**Step 2 到達状態 (Max)**: Score = {best_sw:.6f}")
            with st.expander("Step 2 最適化結果詳細"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Fixed Params:")
                    st.json(vec_to_params(curr_p, kernel))
                with c2:
                    st.write("Best Weights (Optimized):")
                    st.dataframe(pd.DataFrame(curr_w, index=feat_names, columns=["Weight"]).T)
            
            st.divider()
            prog.progress(cy / n_cycles)
            previous_score = best_sw

        st.balloons()
        st.write("## 🏆 Final Result")
        st.write("Best Score:", best_sw)
        
        col1, col2 = st.columns(2)
        with col1:
             st.write("Best Params:")
             st.json(vec_to_params(curr_p, kernel))
        with col2:
             st.write("Best Weights:")
             st.dataframe(pd.DataFrame(curr_w, index=feat_names, columns=["Weight"]))

        # 最終モデルでの評価
        final_X = apply_weights(datas, curr_w)
        final_model_dict = vec_to_params(curr_p, kernel)
        final_model_dict["degree"] = curr_deg
        final_model_dict["max_iter"] = 5000
        final_model_dict["random_state"] = 42
        
        cv_model = SVC(**final_model_dict)
        kf_final = StratifiedKFold(n_splits=k_cv, shuffle=True, random_state=42)
        best_pred = cross_val_predict(cv_model, final_X, labels, cv=kf_final)
        
        st.write("### 混同行列 & 指標")
        cm = confusion_matrix(labels, best_pred, labels=[1, 2, 3])
        st.write(cm)
        
        for i in range(cm.shape[0]):
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            FP = np.sum(cm[:, i]) - TP
            TN = np.sum(cm) - (TP + FN + FP)
            sens = TP / (TP + FN) if (TP + FN) != 0 else 0
            spec = TN / (TN + FP) if (TN + FP) != 0 else 0
            st.write(f"Class {i+1}: Sensitivity={sens*100:.2f}%, Specificity={spec*100:.2f}%")

if __name__ == "__main__":
    main()