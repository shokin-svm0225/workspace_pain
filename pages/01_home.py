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
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu

# ラジオボタンを表示
home_type = st.sidebar.radio("選んでください", ["研究概要","学術講演会", "辞書", "自己紹介"])

if home_type == "研究概要":
    st.title('ホーム')
    st.subheader("研究内容", divider='rainbow')
    st.write("AIモデルの一つであるSVMを用い、整形外科での診断補助システム開発を行う")
    st.subheader("背景", divider='rainbow')
    st.write("整形外科の患者が訴える典型的症状に、疼痛（慢性痛の総称）がある。人口の20％以上が何らかの疼痛を感じているとの報告もある程、普遍的な症状の一つ。  \n医師による疼痛原因の診断法は複数開発されている一方、手法により精度の差があることが知られている。")
    st.subheader("目的", divider='rainbow')
    st.write("人工知能による疼痛診断の自動化を目指す。  \n医学的診断を人工知能で行う際の制約として、「判断根拠を説明できなければいけない」、また「各医療機関が持つデータ量は限定的」というものがある。  \nこれらを解決するため、「少ないデータでも高い精度を達成しやすい」、そして「判断理由の解釈も比較的容易」という特徴を持つ\nSVMを用いることで、高精度かつ説明可能なAIによる診断補助システム構築を目指す。")
    st.subheader("質問項目の説明", divider='rainbow')
    # タブの作成
    tab1, tab2, tab3 = st.tabs(["PainDETECT", "BS-POP", "特徴寮拡大"])
    # 各タブに内容を追加
    with tab1:
        # CSVファイルのパスを指定
        csv_file_path_5 = '質問表/painditect_質問.csv'  # ファイルパスを指定
        df_paindetect = pd.read_csv(csv_file_path_5)
        # データフレームを表示
        st.dataframe(df_paindetect)

    with tab2:
        # CSVファイルのパスを指定
        csv_file_path_6 = '質問表/bspop_質問.csv'  # ファイルパスを指定
        df_bspop = pd.read_csv(csv_file_path_6)
        # データフレームを表示
        st.dataframe(df_bspop)
        # CSVファイルのパスを指定
        csv_file_path_7 = '質問表/bspop_痛みの経過.csv'  # ファイルパスを指定
        df_bspop_answer = pd.read_csv(csv_file_path_7)
        # データフレームを表示
        st.dataframe(df_bspop_answer)

    with tab3:
        st.write("デフォルトの質問項目に新たな質問項目(特徴量)を作成")
        st.write("(似た項目のスコアを掛け合わして新たな特徴量の作成(10/11))")
        st.write("- 痺れ(S1)")
        st.write("--- P8：ピリピリしたり、チクチク刺したりするような感じ（虫が歩いているような、電気が流れているような感じ）がありますか？")
        st.write("--- P12：痛みのある場所に、痺れを感じますか？")
        st.write("- 少しの痛み(S2)")
        st.write("--- P9：痛みがある部位を軽く触れられる（衣服や毛布が触れる）だけでも痛いですか？")
        st.write("--- P13：痛みがある部位を、少しの力（指で押す程度）で押しても痛みが起きますか？")
        st.write("- 機嫌(S3)")
        st.write("--- D4：ちょっとしたことが癪（しゃく）にさわって腹が立ちますか？")
        st.write("--- D14：検査や治療をすすめられたとき、不機嫌、易怒的、または理屈っぽくなる")
        st.write("- しつこさ(S4)")
        st.write("--- D16：病状や手術について繰り返し質問する")
        st.write("--- D18：ちょっとした症状に、これさえなければとこだわる")

    st.subheader("参考サイト", divider='rainbow')
    st.write("- [Streamlit_documentation](https://docs.streamlit.io/): streamlitのドキュメント参考")

    st.subheader("今後の予定", divider='rainbow')
    st.write("- 重みの設定")
    st.write("- カーネルのパラメータの変更・設定（色々なカーネルで試す）")
    st.write("- カーネルの関数設定")
    st.write("- 特徴量エンジニアリング")
    st.write("--- ランダムサーチ")
    st.write("- ハイパーパラメータ(C)のチューニング")
    st.write("- 遺伝的アルゴリズムを用いて、パラメータCを求める")
    st.write("--- クロスバリデーション（交差検証）")
    st.write("--- ランダムサーチ")
    st.write("--- グリッドサーチ")
    st.write("--- ベイズ最適化")
    st.write("- モデルの評価指標の見直し")
    st.write("- Scikit-Learnの学習")

    st.subheader("実験ログ", divider='rainbow')
    st.write("- 20241101 : streamlit上で実験ができるようにしました")
    st.write("- 20241102 : 今まで利用したCSVファイルのカラムを統一にしました")
    st.write("- 20241106 : 各質問項目における相関係数の出力をしました")
    st.write("- 20241107 : streamlit上で相関係数の出力・評価の確認を可能にしました")

elif home_type == "学術講演会":
    st.subheader('学術講演会-予稿', divider='rainbow')
    img = Image.open('picture/home/学術講演会予稿1.png')
    st.image(img, use_container_width=True)
    img = Image.open('picture/home/学術講演会予稿2.png')
    st.image(img, use_container_width=True)

elif home_type == "辞書":
    with st.container(border=True):
        st.subheader('教師あり学習', divider='rainbow')
        st.markdown("""
        入力データと人間があらかじめ付けた正解のラベルに基づき、機械が学習を行い、未知のデータの分類・予測をする
        """)
        img = Image.open('picture/教師あり学習.jpg')
        st.image(img, use_container_width=True)
    with st.container(border=True):
        st.subheader('教師なし学習', divider='rainbow')
        st.markdown("""
        ラベルなしデータ・セットを分析し、類似度や規則性に基いてクラスター（群れ・集団）化する
        """)
        img = Image.open('picture/教師なし学習.jpg')
        st.image(img, use_container_width=True)
    with st.container(border=True):
        st.subheader('特徴量エンジニアリング（データクレンジング）', divider='rainbow')
        st.markdown("""
        破損したデータや不正確なデータを特定し、ドメイン知識などを生かして汎化性能を上げるために新しくデータの特徴量を作成するプロセスのこと。
        - 欠損値の補完
        - 特徴量変換
        - 適切な特徴抽出・特徴量選択
        - 特徴量スケーリング（標準化・正規化）
        """)
    with st.container(border=True):
        st.subheader('k-NN法', divider='rainbow')
        st.markdown("""
        k-NN法（k-nearest neighbor algorithm, k近傍法）は機械学習のアルゴリズムの一つで、データをグループ分けするにあたり、対象データがどのグループに含まれるかを周囲のデータの多数決で推測するという手法である
        - 対象データが近い点を近い順に任意の数、k個だけ選び、その点に最も多く含まれているグループに分類される
        """)
    with st.container(border=True):
        st.subheader('標準化', divider='rainbow')
        st.markdown("""
        標準化(Standardization)は、平均を0、分散を1とするスケーリング手法である。  \n
        （どの特徴量も同じスケールの分布を持つように揃える  例：身長・体重）
        - あるデータX全体の平均をμ、標準偏差をσとする。
        """)
        st.latex(r"""
        z = \frac{x - \mu}{\sigma}
        """)
        st.latex(r"""
        \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2}
        """)
    with st.container(border=True):
        st.subheader('正規化', divider='rainbow')
        st.markdown("""
        正規化(Normalization)は、最小値を0、最大値を1とするスケーリング手法である。
        - あるデータX全体の最大値をX_max、最小値をX_minとする。
        """)
        st.latex(r"""
        x' = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
        """)
        st.markdown("""
        - 注意：外れ値の影響を強く受ける可能性がある
        """)
    with st.container(border=True):
        st.subheader('交差検証(クロスバリデーション)', divider='rainbow')
        st.markdown("""
        交差検証（cross-validation）とは、汎化性能を評価する統計的な手法である。
        また、**過学習**を検出する手法でもある。  \n
        過学習：学習データと非常によく一致するモデルを作成した結果、新しいデータに対して正しい予測ができなくなる状態のこと
        - 例：SVMモデルのパラメータ（C）の調整
          - Cを大きくし過ぎると、線形SVMだとイメージしづらいが、非線形SVMの場合、曲線や曲面と次元数が上がるにつれ
            ほとんどのデータを正しく分類しようと「無理な境界」を引き、汎化性能が低下し、過学習につながる
        """)
    with st.container(border=True):
        st.subheader('SVM（サポートベクトルマシン）', divider='rainbow')
        st.markdown("""
        対象データが近い点を近い順に任意の数、k個だけ選び、その点に最も多く含まれているグループに分類される
                    
        （下の図）
                    
        説明変数（例：色、尻尾の長さ、目の形）とし、クラス◼（例：猫）とクラス▲（例：犬）を分類するとする
        """)
        img = Image.open('picture/home/svm説明03.jpg')
        st.image(img, use_container_width=True)
        st.markdown("""
        境界線を引く時、２つのクラスの真ん中に引けばいいと考えられる。また、真ん中といっても傾きの違う境界線もいくつか挙げられる。
        境界線と各クラスの最も近い点を**サポートベクトル**といい、サパートベクトルと境界線との距離（**マージン**）を最大化するように境界線を決めるアルゴリズムがSVMである。
        （上の図：ハードマージンSVM）
        """)
        with st.container(border=True):
            st.subheader('ソフトマージンSVMとカーネル法', divider='rainbow')
            st.markdown("""
            実際には、上の図のように綺麗にプロットされることはなく、下の２つの図のようなプロットになることが多い。
            線形分離不可のときは、
            1. _ソフトマージンSVM_
            2. _カーネル法_
            という2種類の対処法がある。
            """)
            st.markdown("""
            左の図のように線形分離不可なデータに対して、ペナルティを与えることで誤分類を許して境界線を定めるアルゴリズムを**ソフトマージンSVM**という。
            - 目的関数と制約条件
            """)
            st.latex(r"""
            \begin{align*}
            \min_{w, b, \xi} \quad & \frac{1}{2} \|w\|^2 + C \sum_{i \in \{1 \ldots n\}} \xi_i \\
            \text{s.t.} \quad & y_i(w^\top x_i + b) \geq 1 - \xi_i \quad (i \in \{1 \ldots n\}) \\
            & \xi_i \geq 0
            \end{align*}
            """)
            st.markdown("""
            C : 誤判別をどこまで許容するかを表すパラメータ  \n
            ξ : マージンの内側にはみ出すのを許す度合い  \n
              \n
            右の図のように線形分離不可なデータに対して、データの次元をより高い次元に変換することで線形分類を可能にするアルゴリズムを**カーネル法**という。  \n
            """)
            img = Image.open('picture/home/svm説明01.jpg')
            st.image(img, use_container_width=True)


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
        st.subheader("山登り法", divider='rainbow')
        st.markdown("""
        「今の解よりも良い、今の解に近い解を新しい解にする」ことを繰り返して良い解を求める方法  \n
        （最も代表的な局所探索法として知られている。）
        """)
        img = Image.open('picture/20250523/山登り法.png')
        st.image(img, caption='https://algo-method.com/descriptions/5HVDQLJjbaMvmBL5', use_container_width=True)


elif home_type == "自己紹介":
    st.subheader('自己紹介(2025年4月11日)', divider='rainbow')
    img = Image.open('picture/home/自己紹介01.png')
    st.image(img, use_container_width=True)
    img = Image.open('picture/home/自己紹介02.png')
    st.image(img, use_container_width=True)