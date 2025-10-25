
import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import joblib

# =========================================================
# 🔧 既存のロジックを壊さない UI レイヤ（最小変更）
# =========================================================

def _list_input(label: str, default_text: str, help_text: str = ""):
    txt = st.text_input(label, value=default_text, help=help_text)
    vals = []
    if txt.strip():
        try:
            for x in txt.split(","):
                x = x.strip()
                if any(ch in x for ch in [".", "e", "E"]):
                    vals.append(float(x))
                else:
                    vals.append(int(x))
        except Exception:
            st.warning(f"{label}: 数値に変換できない要素があります。")
    return vals

def _grid_ui():
    st.sidebar.header("探索パラメータ")

    # --- 反復回数（グローバル共有用; 関数が参照するなら使われます） ---
    max_iters = st.sidebar.number_input("ヒルクライミング反復回数", min_value=1, value=200, step=10,
                                        help="関数内でこの値を参照する実装なら、この設定が適用されます。")
    st.session_state["max_iters_ui"] = int(max_iters)

    # --- カーネル選択（最終学習に使用。内部探索がRBF前提でもUIは確保） ---
    kernel_choice = st.sidebar.selectbox("最終モデルのカーネル", ["rbf", "linear", "poly", "sigmoid"], index=0)

    # --- 共通（現行 param_grid: step_size, gamma, C） ---
    st.sidebar.subheader("グリッド（現行：step_size × gamma × C）")

    step_sizes = _list_input(
        "step_size（カンマ区切り）",
        "0.01,0.05",
        "例: 0.005,0.01,0.05"
    )

    # カーネル別パラメータ（UIだけ。現行の探索は gamma と C を使用）
    with st.sidebar.expander("RBF: C / gamma", expanded=True):
        C_values_rbf = _list_input("C（RBF）", "0.1,1,10", "例: 0.1,1,10,100")
        gamma_values_rbf = _list_input("gamma（RBF）", "0.01,0.05,0.1", "例: 0.001,0.01,0.1,1")

    with st.sidebar.expander("linear: C", expanded=False):
        C_values_linear = _list_input("C（linear）", "0.1,1,10", "※ 参考UI（内部ロジック未接続）")

    with st.sidebar.expander("poly: C / gamma / coef0 / degree", expanded=False):
        C_values_poly = _list_input("C（poly）", "0.1,1,10", "")
        gamma_values_poly = _list_input("gamma（poly）", "0.001,0.01,0.1", "")
        coef0_values_poly = _list_input("coef0（poly）", "0.0,0.5,1.0", "")
        degree_values_poly = _list_input("degree（poly）", "2,3,4", "")

    with st.sidebar.expander("sigmoid: C / gamma / coef0", expanded=False):
        C_values_sigmoid = _list_input("C（sigmoid）", "0.1,1,10", "")
        gamma_values_sigmoid = _list_input("gamma（sigmoid）", "0.001,0.01,0.1", "")
        coef0_values_sigmoid = _list_input("coef0（sigmoid）", "0.0,0.5,1.0", "")

    # 現行の param_grid 構築（RBF互換: step_size × gamma × C）
    return {
        "max_iters": int(max_iters),
        "kernel_choice": kernel_choice,
        "step_sizes": step_sizes or [0.01],
        "C_values": C_values_rbf or [0.1, 1],
        "gamma_values": gamma_values_rbf or [0.01, 0.05],
        # 参考：保持しておく（将来の内部対応用）
        "grids_all": {
            "rbf": {"C": C_values_rbf, "gamma": gamma_values_rbf},
            "linear": {"C": C_values_linear},
            "poly": {"C": C_values_poly, "gamma": gamma_values_poly, "coef0": coef0_values_poly, "degree": degree_values_poly},
            "sigmoid": {"C": C_values_sigmoid, "gamma": gamma_values_sigmoid, "coef0": coef0_values_sigmoid},
        }
    }

def apply_weights(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X * w

# =========================================================
# 🛟 フォールバック実装（run_hill_climbing が未定義なら用意）
# =========================================================
try:
    run_hill_climbing  # type: ignore[name-defined]
except NameError:
    def _evaluate(weights_change: np.ndarray,
                  datas: np.ndarray,
                  labels: np.ndarray,
                  C: float,
                  gamma: float,
                  k: int = 5,
                  return_best_split: bool = False):
        X_weighted = datas * weights_change
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = []
        best_fold_score = -np.inf
        best_X_val = best_y_val = best_pred = None

        for train_index, val_index in skf.split(X_weighted, labels):
            X_train, X_val = X_weighted[train_index], X_weighted[val_index]
            y_train, y_val = labels[train_index], labels[val_index]

            model = SVC(C=C, kernel='rbf', gamma=gamma, max_iter=1500)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = float(np.mean(y_pred == y_val))
            scores.append(acc)

            if return_best_split and acc > best_fold_score:
                best_fold_score = acc
                best_X_val, best_y_val, best_pred = X_val, y_val, y_pred

        if return_best_split:
            return float(np.mean(scores)), best_X_val, best_y_val, best_pred
        else:
            return float(np.mean(scores))

    def _hill_climbing(datas: np.ndarray, labels: np.ndarray,
                       C: float, gamma: float,
                       max_iters: int = None,
                       step_size: float = 0.01):
        # 反復回数は UI の値を使う（未設定ならデフォルト200）
        if max_iters is None:
            max_iters = int(st.session_state.get("max_iters_ui", 200))

        n_features = datas.shape[1]
        weights_change = np.ones(n_features, dtype=float)

        best_score, best_X_val, best_y_val, best_pred = _evaluate(
            weights_change, datas, labels, C, gamma, k=5, return_best_split=True
        )
        best_weights = weights_change.copy()
        score_history = [best_score]

        global_best_score = best_score
        global_best_weights = best_weights.copy()
        global_best_pack = (best_X_val, best_y_val, best_pred)

        for _ in range(int(max_iters)):
            improved = False
            for j in range(n_features):
                for delta in (+step_size, -step_size):
                    w_try = best_weights.copy()
                    w_try[j] = max(0.0, w_try[j] + delta)
                    score_try = _evaluate(w_try, datas, labels, C, gamma, k=5, return_best_split=False)
                    if score_try > best_score:
                        best_score = score_try
                        best_weights = w_try
                        improved = True

            score_history.append(best_score)

            if best_score > global_best_score:
                global_best_score = best_score
                global_best_weights = best_weights.copy()
                _, Xv, yv, pv = _evaluate(best_weights, datas, labels, C, gamma, k=5, return_best_split=True)
                global_best_pack = (Xv, yv, pv)

            if not improved:
                break

        return global_best_weights, float(global_best_score), global_best_pack[0], global_best_pack[1], global_best_pack[2], score_history

    def run_hill_climbing(step_size: float, gamma: float, C: float, datas: np.ndarray, labels: np.ndarray):
        """ユーザー既存の呼び出しシグネチャを尊重したフォールバック版。
        返り値のキーも既存コードが期待するものに合わせる。
        """
        weights_best, score, X_val_tmp, y_val_tmp, pred_tmp, score_history = _hill_climbing(
            datas, labels, C=C, gamma=gamma, max_iters=int(st.session_state.get("max_iters_ui", 200)), step_size=step_size
        )
        return {
            "step_size": step_size,
            "gamma": gamma,
            "C": C,
            "score": float(score),
            "weights": [float(f"{w:.4f}") for w in weights_best],
            "weights_raw": np.asarray(weights_best, dtype=float).tolist(),
            "score_history": list(map(float, score_history)),
            "X_val": X_val_tmp,
            "y_val": y_val_tmp,
            "pred": pred_tmp,
        }

# =========================================================
# 🧪 メイン（あなたの既存ブロックを最小編集で内包）
# =========================================================

def main():
    st.title("🧠 Hill Climbing × 並列探索（SVM最適化）")

    # ▼▼▼ ここはあなたの前処理ブロックを想定（コピペで差し替えてください） ▼▼▼
    # 例としてダミーデータを作ります。実運用では df1/df2/df3, edited_df 等から生成してください。
    N = 180
    D = 12
    rng = np.random.default_rng(42)
    df1 = pd.DataFrame(rng.normal(0, 1, size=(N//3, D)), columns=[f"F{i+1}" for i in range(D)])
    df2 = pd.DataFrame(rng.normal(0.5, 1, size=(N//3, D)), columns=[f"F{i+1}" for i in range(D)])
    df3 = pd.DataFrame(rng.normal(-0.5, 1, size=(N - 2*(N//3), D)), columns=[f"F{i+1}" for i in range(D)])

    stocks = df1.columns.tolist()
    edited_df = pd.DataFrame({"columns": stocks, "weights": np.ones(len(stocks))})

    choice_4 = st.selectbox("標準化を行う？", ["する", "しない"], index=0)

    columns = edited_df["columns"].tolist()
    weights = edited_df["weights"].tolist()

    df_nociceptive_train = df1[columns]
    df_neuronociceptive_train = df2[columns]
    df_unknown_train = df3[columns]

    df_nociceptive_train_weighted = df_nociceptive_train.mul(weights, axis=1)
    df_neuronociceptive_train_weighted = df_neuronociceptive_train.mul(weights, axis=1)
    df_unknown_train_weighted = df_unknown_train.mul(weights, axis=1)

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

    if choice_4 == "する":
        scaler = StandardScaler()
        datas = scaler.fit_transform(datas)

    # ▲▲▲ あなたの前処理ここまで（実データに置換） ▲▲▲

    # --- 新しい UI で探索条件を設定（既存ロジックは保持） ---
    ui = _grid_ui()

    # === パラメータ設定 & 実行（あなたの元コードを踏襲） ===
    step_sizes = ui["step_sizes"]
    C_values = ui["C_values"]
    gamma_values = ui["gamma_values"]

    param_grid = [
        (step_size, gamma, C)
        for step_size in step_sizes
        for gamma in gamma_values
        for C in C_values
    ]

    all_results = []
    best_score = 0.0
    best_result = None

    st.write("🔁 並列実行中...")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(run_hill_climbing, step_size, gamma, C, datas, labels):
            (step_size, gamma, C)
            for (step_size, gamma, C) in param_grid
        }
        total_jobs = len(futures)

        done = 0
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)

            if result["score"] > best_score:
                best_score = result["score"]
                best_result = result

            done += 1
            percent = int(done / total_jobs * 100) if total_jobs else 100
            progress_bar.progress(percent, text=f"進捗状況 {done}/{total_jobs}（{percent}％）")
            progress_text.text(f"進捗状況 {done}/{total_jobs}")

    results_df = pd.DataFrame([{
        "step_size": r["step_size"],
        "gamma": r["gamma"],
        "C": r["C"],
        "score": r["score"],
        "weights": r["weights"]
    } for r in all_results])

    elapsed = time.time() - start_time
    st.write(f"⏱ 実行時間: {elapsed:.2f} 秒")

    st.subheader("📊 スコアまとめ")
    st.dataframe(results_df.sort_values(by="score", ascending=False), use_container_width=True)

    st.subheader("📈 一番良かったパラメータのスコア推移")
    if best_result is None:
        st.warning("結果がありません。探索条件を見直してください。")
        return

    best_history = best_result["score_history"]
    fig, ax = plt.subplots()
    ax.plot(range(len(best_history)), best_history)
    ax.set_title("Best Score Progression by Hill Climbing")
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    st.pyplot(fig)

    # === 最終モデルの学習（UI のカーネル選択を適用） ===
    st.subheader("🧩 最終モデルの学習・保存")
    kernel_for_final = ui["kernel_choice"]

    X_weighted_final = apply_weights(datas, np.array(best_result["weights"], dtype=float))

    gamma_final = best_result.get("gamma", None)
    C_final = best_result.get("C", 1.0)
    coef0_final = 0.0
    degree_final = 3

    final_model = SVC(
        C=C_final,
        kernel=kernel_for_final,
        gamma=gamma_final if kernel_for_final in ("rbf", "poly", "sigmoid") else "scale",
        coef0=coef0_final if kernel_for_final in ("poly", "sigmoid") else 0.0,
        degree=degree_final if kernel_for_final == "poly" else 3,
        max_iter=1500
    )
    final_model.fit(X_weighted_final, labels)
    joblib.dump(final_model, "final_model.joblib")
    st.success("✅ 最終モデルを保存しました（final_model.joblib）")

    # === 感度・特異度の計算（CV のベスト fold を使用） ===
    st.subheader("🧮 感度・特異度")
    best_y_val = best_result["y_val"]
    best_pred = best_result["pred"]
    if best_y_val is not None and best_pred is not None:
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
    else:
        st.info("ベストfoldの予測が無いので感度・特異度は表示できません。")

if __name__ == "__main__":
    main()
