import streamlit as st
import itertools
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu

st.title('本日の発表内容')

with st.container(border=True):
    st.subheader("アジェンダ", divider='rainbow')
    st.markdown("""
    - プロスラムの変更
    - 前回の内容
    - 実験の概要
    - 実験結果・考察
    - 今後の予定の確認
    - アドバイス
    """)  

with st.container(border=True):
    st.subheader("プログラムの変更", divider='rainbow')
    st.markdown("""
        - 山登り法とSVMのプログラムを切り分けて再利用可能なコードの実装
        """)  
    with st.container(border=True):
        st.subheader('プログラムの概要', divider='rainbow')
        # タブの作成
        tab1, tab2, tab3 = st.tabs(["山登り法([-ε,0,+ε]の3方向)", "山登り法([-ε,+ε]の2方向)", "SVM(交差検証)"])
        # 各タブに内容を追加
        with tab1:
            body_1 = """
            # 初期重みを [-5, 5) の範囲でランダムに設定（再現性ありなら np.random.seed も併用）
            initial_weights = np.random.randint(-5, 5, datas.shape[1])

            # 山登り法（1つのCに対して最適な重みを探索）
            def hill_climbing(datas, labels, C, initial_weights, max_iter_1=100, step_size=1):
                n_features = datas.shape[1]
                # weights_change = np.ones(n_features)
                weights_change = initial_weights.copy()  # 外から渡された固定の初期重み
                st.write("✅ 初期重み:" + str(weights_change.tolist()))

                best_score, best_X_val, best_y_val, best_pred = evaluate(weights_change, datas, labels, C, return_best_split=True)
                best_weights = weights_change.copy()


                # Streamlitの進捗バーとスコア表示
                hill_bar = st.progress(0)
                score_history = [best_score]


                for i in range(max_iter_1):
                    step_best_score = best_score
                    step_best_weights = weights_change.copy()
                    step_best_X_val, step_best_y_val, step_best_pred = best_X_val, best_y_val, best_pred

                    for idx in range(n_features):
                        for delta in [-step_size, 0, step_size]:
                            trial_weights = weights_change.copy()
                            trial_weights[idx] += delta #idx番目の特徴量だけ delta 分変化させた新しい重みを作成

                            # delta = 0 のときは再評価せず、現在のベストスコアと検証結果をそのまま使用
                            if delta == 0:
                                score = best_score
                                X_val_tmp, y_val_tmp, pred_tmp = best_X_val, best_y_val, best_pred
                            else:
                                score, X_val_tmp, y_val_tmp, pred_tmp = evaluate(
                                    trial_weights, datas, labels, C, return_best_split=True
                                )

                            # 各ステップの中、もっとも良いスコアが得られた場合は、その情報を更新・記録
                            if score > step_best_score:
                                step_best_score = score
                                step_best_weights = trial_weights.copy()
                                step_best_X_val = X_val_tmp
                                step_best_y_val = y_val_tmp
                                step_best_pred = pred_tmp

                    # 一番良かった方向へ重みを更新し、スコアや予測結果を上書き
                    weights_change = step_best_weights
                    best_weights = weights_change.copy()
                    best_score = step_best_score
                    best_X_val, best_y_val, best_pred = step_best_X_val, step_best_y_val, step_best_pred

                    # スコア履歴に今回のベストスコアを追加
                    score_history.append(best_score)
                    percent = int((i + 1) / max_iter_1 * 100)
                    hill_bar.progress(percent, text=f"進捗状況{percent}%")

                return best_weights, best_score, best_X_val, best_y_val, best_pred, score_history
            """
            st.code(body_1, language="python")

        with tab2:
            body_2 = """
            # 初期重みを [-5, 5) の範囲でランダムに設定（再現性ありなら np.random.seed も併用）
            initial_weights = np.random.randint(-5, 5, datas.shape[1])

            # 山登り法で、1つのCに対して最適な特徴量の重みベクトルを探索
            def hill_climbing(datas, labels, C, initial_weights, max_iter_1=100, step_size=0.1):
                n_features = datas.shape[1]

                # 初期重みをコピー（元の初期重みは他のC値でも使えるように保持）
                weights_change = initial_weights.copy()
                st.write("✅ 初期重み:" + str(weights_change.tolist()))

                # 初期重みに対するスコアと検証データの情報を取得
                best_score, best_X_val, best_y_val, best_pred = evaluate(
                    weights_change, datas, labels, C, return_best_split=True
                )
                best_weights = weights_change.copy()

                # Streamlit の進捗バーを初期化
                hill_bar = st.progress(0)
                score_history = [best_score]  # スコアの履歴を保存

                # hill climbing を max_iter_1 回繰り返す
                for i in range(max_iter_1):
                    # 各ステップで最良のスコアを探す
                    step_best_score = best_score
                    step_best_weights = weights_change.copy()
                    step_best_X_val, step_best_y_val, step_best_pred = best_X_val, best_y_val, best_pred

                    # 各特徴量に対して ±step_size 変更を試す
                    for idx in range(n_features):
                        for delta in [-step_size, step_size]:
                            trial_weights = weights_change.copy()  # 現在の重みをコピー
                            trial_weights[idx] += delta            # 一つの特徴量だけを変更

                            # 新しい重みでモデルを評価
                            score, X_val_tmp, y_val_tmp, pred_tmp = evaluate(
                                trial_weights, datas, labels, C, return_best_split=True
                            )

                            # スコアが改善されたら、その重みを保存
                            if score > step_best_score:
                                step_best_score = score
                                step_best_weights = trial_weights.copy()
                                step_best_X_val = X_val_tmp
                                step_best_y_val = y_val_tmp
                                step_best_pred = pred_tmp

                    # ステップ内で最良だった重みを採用（更新）
                    weights_change = step_best_weights
                    best_weights = weights_change.copy()
                    best_score = step_best_score
                    best_X_val, best_y_val, best_pred = step_best_X_val, step_best_y_val, step_best_pred

                    # スコア履歴の更新と進捗表示
                    score_history.append(best_score)
                    percent = int((i + 1) / max_iter_1 * 100)
                    hill_bar.progress(percent, text=f"進捗状況{percent}%")

                # 最終的に見つかった最良の重みとスコア、検証結果、スコアの推移を返す
                return best_weights, best_score, best_X_val, best_y_val, best_pred, score_history
            """
            st.code(body_2, language="python")

        with tab3:
            body_3 = """
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
            """
            st.code(body_3, language="python")

        
with st.container(border=True):
    st.subheader("実験の概要", divider='rainbow')
    st.markdown("""
        各質問項目に対して適切な重み付けを山登り法で行う  \n
        - 各ステップで特徴量すべてに対して[-ε,0,+ε]の3方向で評価、重みの更新を行う
          - 初期値：全て1,乱数
        - 各ステップで特徴量すべてに対して[-ε,+ε]の2方向で評価、重みの更新を行う
          - 初期値：全て1,乱数
        **準備**
        - 質問表：BS-POP
        - 欠損値の補完：欠損値削除
        - 重み付けの初期値
          - 全て1.0
          - 全て乱数
            - 山登り法は初期値によって局所最適解を求めてしまう可能性があるから
        - 試行回数：100
        - 重み更新の大きさ：0.1,1
        - カーネル：線形カーネル
        - 評価：5-分割交差検証(random_state=42)
        - パラメータチューニング(C)：グリッドサーチ
        - 結果の出力：正答率（平均）, 感度 , 特異度
        """)
    with st.container(border=True):
        st.subheader("山登り法", divider='rainbow')
        st.markdown("""
        「今の解よりも良い、今の解に近い解を新しい解にする」ことを繰り返して良い解を求める方法  \n
        （最も代表的な局所探索法として知られている。）
        """)
        img = Image.open('picture/20250523/山登り法.png')
        st.image(img, caption='https://algo-method.com/descriptions/5HVDQLJjbaMvmBL5', use_container_width=True)

with st.container(border=True):
    st.subheader("実験結果", divider='rainbow')
    st.markdown("""
    - [-ε,0,+ε]の3方向
    """)
    img1 = Image.open('picture/20250613/スクリーンショット 2025-06-13 7.56.53.png')
    img2 = Image.open('picture/20250613/0.1重み初期1.png')
    img3 = Image.open('picture/20250613/1初期1_100.png')
    img4 = Image.open('picture/20250613/third0.1.png')
    img5 = Image.open('picture/20250613/third1_100.png')
    st.image(img1, caption='欠損値削除', use_container_width=True)
    # カラムを3つ作成
    col1, col2 = st.columns(2)
    # 各カラムに画像を表示
    with col1:
        st.image(img2, caption="初期値：全1,更新の大きさ：0.1", use_container_width=True)
    with col2:
        st.image(img3, caption="初期値：全1,更新の大きさ：1", use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.image(img4, caption="初期値：乱数,更新の大きさ：0.1", use_container_width=True)
    with col4:
        st.image(img5, caption="初期値：乱数,更新の大きさ：1", use_container_width=True)
    st.markdown("""
    - [-ε,+ε]の2方向
    """)
    img1 = Image.open('picture/20250613/スクリーンショット 2025-06-13 7.57.46.png')
    img2 = Image.open('picture/20250613/second全1_100_0.1.png')
    img3 = Image.open('picture/20250613/second全1.0100_1.png')
    img4 = Image.open('picture/20250613/second100_0.01.png')
    img5 = Image.open('picture/20250613/second100_1.png')
    st.image(img1, caption='欠損値削除', use_container_width=True)
    # カラムを3つ作成
    col1, col2 = st.columns(2)
    # 各カラムに画像を表示
    with col1:
        st.image(img2, caption="初期値：全1,更新の大きさ：0.1", use_container_width=True)
    with col2:
        st.image(img3, caption="初期値：全1,更新の大きさ：1", use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.image(img4, caption="初期値：乱数,更新の大きさ：0.1", use_container_width=True)
    with col4:
        st.image(img5, caption="初期値：乱数,更新の大きさ：1", use_container_width=True)

with st.container(border=True):
    st.subheader("結果・考察", divider='rainbow')
    st.markdown("""
    - 精度としては以前とあまり変わらず、65%前後で落ち着いている
    - 初期値や更新の大きさを変えてもどれも数回の試行でスコアが変わらなくなってしまった（特に[-ε,+ε]の2方向）
      - 原因：交差検証のrandom_stateが固定されているため、少し重みをずらしてもスコアが変わらない・プログラムの間違い
    - [-ε,+ε]の2方向の場合は、各ステップごとにベストスコアを評価し、必ず「どれか一番良い方向」に更新するため、変わらなくなるのはおかしい？
    """)

with st.container(border=True):
    st.subheader("今後の予定の確認", divider='rainbow')
    st.markdown("""
    - 山登り法
      - PainDITECTとFUSIONの質問表でも実装
      - 試行回数を増やして実験
      - 乱数の範囲を変えて実験
    - 遺伝的アルゴリズムによる重み付け
    - 侵害需要性疼痛と診断されたデータと神経障害性疼痛と診断されたデータの回答傾向を可視化
    """)