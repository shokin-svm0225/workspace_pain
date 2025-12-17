import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib
import streamlit as st
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# =========================
# 定数設定（最大値の制限）
# =========================
MAX_GAMMA = 10.0
MAX_COEF0 = 10.0

# =========================
# ユーティリティ & ヘルパー関数
# =========================

def parse_num_list(s, dtype=float):
    """カンマ区切り文字列をリストに変換"""
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

def apply_weights(datas, weights):
    """データに重みを適用（重みは固定値）"""
    return datas * weights

def vec_to_params(vec, kernel):
    """探索用ベクトルをSVMのパラメータ辞書に変換する"""
    p_dict = {"kernel": kernel}
    idx = 0
    p_dict["C"] = vec[idx]
    idx += 1
    if kernel in ["rbf", "poly", "sigmoid"]:
        if idx < len(vec):
            p_dict["gamma"] = vec[idx]
            idx += 1
    if kernel in ["poly", "sigmoid"]:
        if idx < len(vec):
            p_dict["coef0"] = vec[idx]
            idx += 1
    return p_dict

def random_jump_params(params, k, strength, param_types, step_sizes, rng=None):
    """
    パラメータ用のランダムジャンプ（coef0非負制約 & 加算ジャンプ & 上限ガード対応版）
    Args:
        step_sizes: 各パラメータの山登り法でのステップ幅
    """
    rng = np.random.default_rng() if rng is None else rng
    p = np.asarray(params, dtype=float).copy()
    d = len(p)
    k = max(1, min(int(np.ceil(k)), d))
    
    sel = rng.choice(np.arange(d), size=k, replace=False)
    
    for idx in sel:
        if param_types[idx] == 'log':
            # === logスケール (C, gamma) は倍率 ===
            noise = rng.uniform(strength[0], strength[1])
            p[idx] *= noise
            p[idx] = max(p[idx], 0.0001) # 下限 (正の値)
            
            # Gammaの上限ガード (idx=1 が gamma であると仮定)
            if idx == 1: 
                p[idx] = min(p[idx], MAX_GAMMA)

        else:
            # === linearスケール (coef0) は加算 ===
            scale_factor = 5.0 if strength[1] < 2.0 else 20.0
            
            # step_sizesから現在のステップ幅を取得
            current_step = step_sizes[idx]
            
            jump_val = rng.uniform(-scale_factor, scale_factor) * current_step
            p[idx] += jump_val
            
            # Coef0の下限(0)と上限ガード
            p[idx] = max(p[idx], 0.0)
            if idx == 2: # idx=2 が coef0 であると仮定
                p[idx] = min(p[idx], MAX_COEF0)
                
    return p

# =========================
# 評価関数 & 最適化ロジック
# =========================

def evaluate_svm_optim(svm_param_vec, fixed_weights, datas, labels, kernel, degree, k=5, return_best_split=False, max_iter_svc=1500):
    """SVMパラメータの評価関数"""
    X_weighted = apply_weights(datas, fixed_weights)
    params = vec_to_params(svm_param_vec, kernel)
    params["degree"] = degree
    params["max_iter"] = max_iter_svc
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    best_fold_score = -np.inf
    best_X_val, best_y_val, best_pred = None, None, None

    for train_index, val_index in skf.split(X_weighted, labels):
        X_train, X_val = X_weighted[train_index], X_weighted[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        model = SVC(**params)
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

def hill_climbing_svm_params(datas, labels, kernel, degree, fixed_weights, init_params_vec, step_sizes_vec, max_iter_hc=1000, k=5, max_iter_svc=1500, stagnate_L=20, small_strength=(0.85, 1.15), big_strength=(0.4, 2.5)):
    """SVMパラメータに対する山登り法"""
    rng = np.random.default_rng()
    current_params = np.array(init_params_vec, dtype=float)
    n_params = len(current_params)
    
    # パラメータの種類を判定 (順序: C -> gamma -> coef0)
    param_types = []
    idx = 0
    param_types.append('log') # C (idx=0)
    idx += 1
    if kernel in ["rbf", "poly", "sigmoid"]:
        if idx < n_params:
            param_types.append('log') # gamma (idx=1)
            idx += 1
    if kernel in ["poly", "sigmoid"]:
        if idx < n_params:
            param_types.append('linear') # coef0 (idx=2)
            idx += 1
    
    best_score, best_X_val, best_y_val, best_pred = evaluate_svm_optim(
        current_params, fixed_weights, datas, labels, kernel, degree, k, True, max_iter_svc
    )
    best_params = current_params.copy()
    score_history = [best_score]
    
    global_best_score = best_score
    global_best_params = best_params.copy()
    global_best_pack = (best_X_val, best_y_val, best_pred)
    
    no_improve = 0
    
    for _ in range(max_iter_hc):
        step_best_score = -np.inf
        candidates = []
        
        for idx in range(n_params):
            step_val = step_sizes_vec[idx]
            
            if param_types[idx] == 'log':
                # 対数スケール（倍率）
                vals_to_try = [
                    best_params[idx] * step_val,
                    best_params[idx] / step_val
                ]
            else:
                # 線形スケール（加減算）
                vals_to_try = [
                    best_params[idx] + step_val,
                    best_params[idx] - step_val
                ]

            for val in vals_to_try:
                trial_params = best_params.copy()
                trial_params[idx] = val
                
                # === 制約処理 (下限 & 上限) ===
                if param_types[idx] == 'log':
                     trial_params[idx] = max(trial_params[idx], 0.0001)
                     # Gamma (idx=1) の上限
                     if idx == 1:
                         trial_params[idx] = min(trial_params[idx], MAX_GAMMA)
                else:
                     # Coef0 (idx=2) の非負制約 & 上限
                     trial_params[idx] = max(trial_params[idx], 0.0)
                     if idx == 2:
                         trial_params[idx] = min(trial_params[idx], MAX_COEF0)
                
                score, Xv, yv, pr = evaluate_svm_optim(
                    trial_params, fixed_weights, datas, labels, kernel, degree, k, True, max_iter_svc
                )
                
                if score > step_best_score:
                    step_best_score = score
                    candidates = [(trial_params, Xv, yv, pr)]
                elif score == step_best_score:
                    candidates.append((trial_params, Xv, yv, pr))
        
        if step_best_score >= best_score:
            sel_p, sel_Xv, sel_yv, sel_pr = random.choice(candidates)
            best_params = sel_p
            best_score = step_best_score
            best_X_val, best_y_val, best_pred = sel_Xv, sel_yv, sel_pr
            
            if best_score > global_best_score:
                global_best_score = best_score
                global_best_params = best_params.copy()
                global_best_pack = (best_X_val, best_y_val, best_pred)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        # === 停滞時のジャンプ処理 ===
        if no_improve >= stagnate_L:
            
            # ★小ジャンプでも大ジャンプでも、常に「全パラメータ」を動かす
            jump_k = n_params 
            
            if no_improve >= stagnate_L * 2:
                # === 大ジャンプ (強度: big, 個数: 全部) ===
                best_params = random_jump_params(
                    best_params, k=jump_k, strength=big_strength, 
                    param_types=param_types, step_sizes=step_sizes_vec, rng=rng
                )
                no_improve = 0 # リセット
            else:
                # === 小ジャンプ (強度: small, 個数: 全部) ===
                best_params = random_jump_params(
                    best_params, k=jump_k, strength=small_strength, 
                    param_types=param_types, step_sizes=step_sizes_vec, rng=rng
                )
        
        score_history.append(best_score)
        
    return global_best_params, global_best_score, global_best_pack, score_history

def run_hill_svm_wrapper(kernel, degree, fixed_weights, init_params, step_sizes, datas, labels, max_iter_hc, k_cv, max_iter_svc, stagnate_L):
    """並列処理用のラッパー関数"""
    best_params, score, pack, history = hill_climbing_svm_params(
        datas, labels, kernel, degree, fixed_weights, init_params, step_sizes,
        max_iter_hc=max_iter_hc, k=k_cv, max_iter_svc=max_iter_svc, stagnate_L=stagnate_L
    )
    return {
        "kernel": kernel,
        "degree": degree,
        "best_params_vec": best_params,
        "score": score,
        "history": history,
        "pack": pack
    }

# =========================
# UI & メイン処理
# =========================

if __name__ == "__main__":
    st.sidebar.header("SVMパラメータ最適化設定")

    # 1. アルゴリズム選択
    kernel = st.sidebar.selectbox("SVMカーネル", ["linear", "rbf", "poly", "sigmoid"], index=1)

    # ★★★ 初期値候補の設定（UI追加） ★★★
    st.sidebar.markdown("---")
    st.sidebar.markdown("**初期値候補（マルチスタート用）**")
    st.sidebar.caption("ワーカーごとにこのリストからランダムに初期値を選びます。カンマ区切りで入力してください。")
    
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


    st.sidebar.markdown("---")
    st.sidebar.markdown("**山登り法のステップ幅設定**")
    
    step_C = st.sidebar.number_input("Step Rate (C) ※倍率", value=1.5, step=0.1, help="2.0なら 1→2→4 または 1→0.5→0.25")

    step_gamma = 1.5
    step_coef0 = 0.1

    if kernel in ["rbf", "poly", "sigmoid"]:
        step_gamma = st.sidebar.number_input("Step Rate (gamma) ※倍率", value=1.5, step=0.1, format="%.2f")
    if kernel in ["poly", "sigmoid"]:
        step_coef0 = st.sidebar.number_input("Step Size (coef0) ※加算", value=0.5, step=0.05)

    st.sidebar.markdown("---")
    max_iter_hc = st.sidebar.number_input("山登り法の反復回数", min_value=10, max_value=5000, value=1000, step=50)
    
    # 停滞判定のUI入力を変数に格納する（バグ修正済み）
    stagnate_L = st.sidebar.number_input("停滞判定ステップ L", min_value=5, max_value=200, value=10)
    
    k_cv = st.sidebar.slider("CV分割数 (k)", 2, 10, 5)

    st.sidebar.markdown("---")
    n_components = st.sidebar.slider("PCA主成分数", 2, 20, 5)
    
    # 並列数
    st.sidebar.markdown("---")
    n_workers = st.sidebar.slider("並列ワーカー数（スタート地点の数）", 1, 8, 4, help="探索を開始する初期地点の数です。")

    # ==== データセット設定 (既存コード維持) ====
    st.sidebar.header("データセット設定")
    options_miss = ['欠損値データ削除', '中央値補完', '平均値補完', 'k-NN法補完']
    choice_1 = st.sidebar.selectbox('欠損値の対応', options_miss, index=None, placeholder="選択してください")

    options_sheet = ['PainDITECT', 'BS-POP', 'FUSION']
    choice_2 = st.sidebar.selectbox('使用する質問表', options_sheet, index=None, placeholder="選択してください")

    # -- データ読み込み --
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
    else:
        st.stop()

    # --- データ前処理 ---
    options_std = ['する', 'しない']
    choice_4 = st.sidebar.selectbox('データの標準化', options_std, index=None, placeholder="選択してください")

    if choice_4 is None:
        st.stop()

    if choice_4 == "する":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    # PCA
    pca = PCA(n_components, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)
    pca_cols = [f"PCA{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df1.index)

    # 結合
    df_pca_final = pd.concat([df1[[pain_col]], df_pca], axis=1)
    feature_names = pca_cols

    st.success("PCA 実行完了")

    # ==== 重み設定 ====
    st.sidebar.markdown("### 特徴量の重み (固定)")
    if "weights" not in st.session_state:
        st.session_state.weights = {col: 1.0 for col in feature_names}

    if st.button("重みをリセット", key="weights_reset"):
        for col in feature_names:
            st.session_state.weights[col] = 1.0
        st.success("全ての重みを1.0にリセットしました")

    weights_list = []
    for col in feature_names:
        val = st.sidebar.slider(f"{col} weight", -5.0, 5.0, st.session_state.weights.get(col, 1.0), 0.1, key=f"w_{col}")
        st.session_state.weights[col] = val
        weights_list.append(val)
        
    fixed_weights = np.array(weights_list)

    st.info("💡 上記のスライダーで設定した重みは「固定」され、SVMのパラメータのみを最適化します。")

    # =========================
    # 実行ボタン
    # =========================
    if st.button("SVMパラメータ最適化を開始"):
        
        # 1. データ準備
        df_nociceptive = df_pca_final[df_pca_final[pain_col] == "侵害受容性疼痛"]
        df_neuropathic = df_pca_final[df_pca_final[pain_col] == "神経障害性疼痛"]
        df_other = df_pca_final[~df_pca_final[pain_col].isin(["侵害受容性疼痛", "神経障害性疼痛"])]
        
        X1 = df_nociceptive[feature_names].values
        X2 = df_neuropathic[feature_names].values
        X3 = df_other[feature_names].values
        
        datas = np.vstack([X1, X2, X3]).astype(np.float32)
        
        l1 = np.full(len(X1), 1, dtype=int)
        l2 = np.full(len(X2), 2, dtype=int)
        l3 = np.full(len(X3), 3, dtype=int)
        labels = np.concatenate([l1, l2, l3])

        st.title("🧠 SVMハイパーパラメータ最適化")
        st.write(f"カーネル: **{kernel}**")
        st.write("探索方式: **Degree(次数)は総当たり、C/Gamma等は山登り法（マルチスタート）** で並列探索します。")

        # 2. パラメータ初期値の候補（UIから取得）
        step_vec = [step_C]
        param_names = ["C"]
        
        if kernel in ["rbf", "poly", "sigmoid"]:
            step_vec.append(step_gamma)
            param_names.append("gamma")
        
        if kernel in ["poly", "sigmoid"]:
            step_vec.append(step_coef0)
            param_names.append("coef0")
            
        # 3. 並列タスクの生成
        if kernel == "poly":
            candidate_degrees = [2, 3]
        else:
            candidate_degrees = [3] # rbf等ではダミー
            
        futures_input = []
        
        import itertools
        cycle_degrees = itertools.cycle(candidate_degrees)
        
        for _ in range(n_workers):
            d = next(cycle_degrees)
            
            # ★★★ ランダムな初期位置の選択（ユーザー指定リストから選ぶ） ★★★
            start_C = random.choice(candidates_C)
            
            this_init_list = [start_C]
            
            if kernel in ["rbf", "poly", "sigmoid"]:
                start_gamma = random.choice(candidates_gamma)
                this_init_list.append(start_gamma)
                
            if kernel in ["poly", "sigmoid"]:
                start_coef0 = random.choice(candidates_coef0)
                this_init_list.append(start_coef0)
            
            this_init = np.array(this_init_list)
            
            # 初期値に微小ノイズを加える (log/linear共通で簡易的に倍率ノイズ)
            this_init = this_init * np.random.uniform(0.8, 1.2, len(this_init))
            
            # 絶対値ガード (初期値生成時)
            this_init[0] = max(this_init[0], 0.0001) # C
            if kernel in ["rbf", "poly", "sigmoid"]:
                this_init[1] = max(this_init[1], 0.0001) # Gamma
                this_init[1] = min(this_init[1], MAX_GAMMA) # 上限ガード
            if kernel in ["poly", "sigmoid"]:
                # Coef0は3番目(index=2)
                this_init[2] = max(this_init[2], 0.0) # 下限
                this_init[2] = min(this_init[2], MAX_COEF0) # 上限ガード
            
            futures_input.append({
                "degree": d,
                "init": this_init
            })

        st.write(f"探索パラメータ: {param_names}")
        st.write(f"探索Degree候補: {candidate_degrees if kernel=='poly' else '-(固定)'}")
        st.info(f"並列処理におけるスタート地点の数 {n_workers} 個が、指定された初期値リストからランダムに選んで山登りを開始します。")
        
        # 4. 並列実行 (joblib)
        best_overall_score = -np.inf
        best_overall_result = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_tasks = len(futures_input)
        
        results_generator = Parallel(n_jobs=n_workers, return_as="generator")(
            delayed(run_hill_svm_wrapper)(
                kernel,
                inp["degree"],
                fixed_weights,
                inp["init"],
                step_vec,
                datas, labels,
                max_iter_hc, k_cv, 1500, stagnate_L
            ) for inp in futures_input
        )
        
        completed_count = 0
        for res in results_generator:
            completed_count += 1
            progress_bar.progress(completed_count / total_tasks)
            
            if res["score"] > best_overall_score:
                best_overall_score = res["score"]
                best_overall_result = res
                
                degree_msg = f"(Degree={res['degree']})" if kernel == "poly" else ""
                status_text.write(f"暫定1位更新: Score={best_overall_score:.4f} {degree_msg}")
        
        # 5. 結果表示
        st.success("探索完了！")
        st.markdown(f"### 🏆 最高正答率: **{best_overall_score*100:.2f}%**")
        
        best_vec = best_overall_result["best_params_vec"]
        best_degree = best_overall_result["degree"]
        
        # パラメータ復元
        final_params = vec_to_params(best_vec, kernel)

        if kernel == "poly":
            final_params["degree"] = best_degree
        else:
            final_params["degree"] = "なし"
        
        col1, col2 = st.columns(2)
        with col1:
            st.json(final_params)
        with col2:
            st.write("最終重み")
            st.dataframe(pd.DataFrame([fixed_weights], columns=feature_names))

        # 履歴グラフ
        fig, ax = plt.subplots()
        ax.plot(best_overall_result["history"], label="Score History")
        ax.set_title("Optimization History (Best Thread)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)
        
        # モデル保存
        final_X = apply_weights(datas, fixed_weights)
        
        # モデル構築用
        model_params = vec_to_params(best_vec, kernel)
        model_params["degree"] = best_degree
        
        final_model = SVC(**model_params)
        final_model.fit(final_X, labels)
        joblib.dump(final_model, "optimized_svm_model.joblib")
        st.success("モデルを 'optimized_svm_model.joblib' に保存しました。")

        # 評価用データの取得
        best_y_val = best_overall_result["pack"][1]
        best_pred = best_overall_result["pack"][2]

        # 感度・特異度の計算のために混同行列自体は内部で作る
        cm = confusion_matrix(best_y_val, best_pred, labels=[1, 2, 3])

        # クラスごとの指標計算と表示
        n_classes = cm.shape[0]
        for i in range(n_classes):
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            FP = np.sum(cm[:, i]) - TP
            TN = np.sum(cm) - (TP + FN + FP)

            # ゼロ除算回避
            sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

            # 結果を表示
            st.write(f"疼痛 {i+1}: 感度 = {sensitivity * 100:.2f}%, 特異度 = {specificity * 100:.2f}%")