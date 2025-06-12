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
    - 実験結果
    - 考察
    - 今後の予定の確認
    - アドバイス
    """)  

with st.container(border=True):
    st.subheader("プログラムの変更", divider='rainbow')
    st.markdown("""
        - SVMのライブラリをOpenCVからscikit-learnに変更
        """)  
    with st.container(border=True):
        st.subheader('scikit-learn', divider='rainbow')
        st.markdown("""
        Python で利用できるデータ分析や機械学習のためのライブラリの一つ

        - 機械学習のプロジェクト全体を一つのライブラリで管理することが可能
            - データの前処理、教師あり学習、教師なし学習、モデル選択、評価など
        - 非常に充実したドキュメンテーションがある
            - [Scikit-learn_documentation](https://scikit-learn.org/stable/user_guide.html): scikit-learnのドキュメント参考")
        """)
        st.markdown("**SVM（サポートベクトルマシン）**")
        body_1 = """
        class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', 
            coef0=0.0, shrinking=True, probability=False, tol=0.001, 
            cache_size=200, class_weight=None, verbose=False, max_iter=- 1, 
            decision_function_shape='ovr', break_ties=False, random_state=None)
        """
        st.code(body_1, language="python")
        st.markdown("**特徴量X, クラスyを学習データとして学習する**")
        body_2 = """
        fit(X,y)
        """
        st.code(body_2, language="python")
        st.markdown("**テストデータXに対するクラスの予測結果を出力する**")
        body_3 = """
        predict(X)
        """
        st.code(body_3, language="python")
        st.markdown("**K-分割交差検証**")
        body_4 = """
        class sklearn.model_selection.StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)
        """
        st.code(body_4, language="python")
        st.markdown("**標準化**")
        body_5 = """
        class sklearn.preprocessing.StandardScaler(*, copy=True, with_mean=True, with_std=True)
        """
        st.code(body_5, language="python")
        st.markdown("**データを学習用とテスト用に分割する**")
        body_6 = """
        sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
        """
        st.code(body_6, language="python")
        
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
    - 欠損値削除
    """)
    img = Image.open('picture/20250523/20250523_欠損値削除.png')
    st.image(img, caption='欠損値削除', use_container_width=True)
    st.markdown("""
    - 中央値補完
    """)
    img = Image.open('picture/20250523/20250523_中央値.png')
    st.image(img, caption='中央値補完', use_container_width=True)
    st.markdown("""
    - 平均値補完
    最初に34.62%と比較的大きい値が出てしまい、その後更新なし
    """)
    img = Image.open('picture/20250523/20250523平均値.png')
    st.image(img, caption='平均値補完', use_container_width=True)
    st.markdown("""
    - k-NN法補完   
    """)
    img = Image.open('picture/20250523/20250523_knn.png')
    st.image(img, caption='k-NN法補完', use_container_width=True)

with st.container(border=True):
    st.subheader("結果・考察", divider='rainbow')
    st.markdown("""
    - 交差検証で評価をしているものの何回か実行すると離れた結果になる
      - 平均値補完では、最初に34.62%と比較的大きい値が出てしまい、その後更新なし
    - 主観で重み付けを行うのは厳しいかな
    - 制度に差があるのは、データ数が少ないから？
    - 侵害需要性疼痛と診断されたデータと神経障害性疼痛と診断されたデータの回答傾向を可視化し、神経障害性疼痛をうまく識別する特徴量が何かを探して重み付けをするのはどうか
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