import streamlit as st
import itertools
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from streamlit_option_menu import option_menu
ImageFile.LOAD_TRUNCATED_IMAGES = True

st.title('本日の発表内容')

with st.container(border=True):
    st.subheader("目次", divider='rainbow')
    st.markdown("""
    - 実験1
    - 実験2
    - 今後の予定の確認
    - アドバイス
    """)

with st.container(border=True):
    st.subheader("実験1", divider='rainbow')
    st.markdown("""
    - 変更点1
      - モジュールの変更：ThreadPoolExecutor → ProcessPoolExecutor\n
    ***並列タスク実行***\n
    concurrent.futures モジュール：並列処理を行う仕組みとして、マルチスレッドによる並列化を行う
    - ThreadPoolExecutor：並行処理(複数のタスクを同時に進めているように見せる仕組み)
    - ProcessPoolExecutor：並列処理(複数のCPUコアやプロセッサが、同時に複数のタスクを実際に実行すること)
    """)
    code = '''
    #モジュール 
    class concurrent.futures.Executor
    #ThreadPoolExecutor
    class concurrent.futures.ThreadPoolExecutor(max_workers=None, thread_name_prefix='', initializer=None, initargs=())
    #ProcessPoolExecutor
    class concurrent.futures.ProcessPoolExecutor(max_workers=None, mp_context=None, initializer=None, initargs=())'''
    st.code(code, language='python')

    st.markdown("""
    - 変更点2
      - 並列処理：外側（グリッドサーチ）+内側（山登り法）の二重並列 \n
        → 外側（グリッドサーチ）のみ \n
    ***詳細***\n
    これまで外側（グリッドサーチ）+内側（山登り法）の二重並列で実装を行なってきたが、外側(|C| × |gamma| × |degree| × |coef0|通り) × 内側(2 × 特徴量数 × 山登りの回数)
    でコア数以上にプロセスが増えるとオーバーヘッドが大きくなり、逆に遅くなる可能性がある.\n
    そのため、並列処理を外側（グリッドサーチ）のみで行うことにする.
    """)

    st.markdown("""
    - 実験・検証
    二重並列・外側のみ・内側のみの3つの計算時間の比較を行う
      - 質問表：FUSION(質問項目:31)
      - 特徴量エンジニアリング：欠損値削除、平均 0・分散 1に標準化
      - 分類モデル：SVM
      - カーネル：線形カーネル
      - 性能評価：5分割交差検証
      - パラメータチューニング（C）：0.1固定
      - 各特徴量に対して重み付け (x′ = α × x )：山登り法（初期値：1，ステップ幅：0.01，反復回数：30)\n
    - 結果・考察
      - 二重並列：54.02秒
      - 外側のみ：10.36秒
      - 内側のみ：52.63秒 \n
    外側のみにすると、約40秒も速くなることがわかり、プロセスが増えすぎてしまい、オーバーヘッドが大きくなってしまっていた可能性があった.
    """)

        
with st.container(border=True):
    st.subheader("実験2", divider='rainbow')
    st.markdown("""
        **主成分分析とは** \n
        多くの変数を持つ複雑なデータを、少数の指標に要約する統計手法である.
        - 2変数や3変数程度であれば，視覚的にデータの分布や傾向を理解することができる.しかし，変数の数が多くなると，各変数間の関係を直感的に把握することは困難となる．
        - 例：学校の成績、体重身長
        - 主成分分析のアルゴリズム：まず、各変数の影響度をそろえるためにデータを標準化（平均0、標準偏差1）し、変数間の関係性を示す共分散行列を算出する.
                この行列の固有値と固有ベクトルを計算すると、固有値の大きい順に主成分が決定される. \n
        **課題**
        - PainDETECT(質問項目:13)，BS-POP(質問項目:18)，FUSION(質問項目:31)と多次元の特徴量空間を扱っている.
        - 似ている質問項目がある.
        - 特徴量が多すぎて重み付けが悪い方向に行っているのではないか\n
        **目的** 
        - SVM分類における次元削減
        - 次元削減しても高い制度を保つことができるか
        - :red[計算時間の短縮]\n
        **検証** \n
        下記に記載している実験概要の内容で、PCAを導入して精度・計算時間の比較・検証を行う.
        - 線形カーネルのみ
        - 主成分数(80%)： 7次元(PainDETECT)、11次元(BS-POP)、18次元(FUSION)\n 
        **実験概要**\n
          - 質問表：PainDETECT(質問項目:13)，BS-POP(質問項目:18)，及び両者を結合したFUSION(質問項目:31)
          - 特徴量エンジニアリング：欠損値削除、平均 0・分散 1に標準化
          - 分類モデル：SVM
          - カーネル：線形カーネル，多項式カーネル
          - 性能評価：5分割交差検証
          - パラメータチューニング（C,γ,d,c）：グリッドサーチ
            - 線形カーネル：C(0.1,1) 2通り
            - 多項式カーネル：C(0.1,1),gamma(0.01),degree(2,3),coef0(3,5,8) 12通り
          - 各特徴量に対して重み付け (x′ = α × x )：山登り法（初期値：1，ステップ幅：0.01，反復回数：100)\n
        """)
    
    img = Image.open('picture/20251107/pca_change.png')
    st.image(img, use_container_width=True)

    st.markdown("""
        ***結果・考察***
        - 主成分分析を行うことで、80%の累積寄与率で約40%の次元削減で計算時間を約半分に短縮することができた
        - 主成分分析で次元削減しても高いスコアを維持することができた
        """)

with st.container(border=True):
    st.subheader("ランダムジャンプを導入した山登り法による考察", divider='rainbow')
    st.markdown("""
    **ランダムジャンプを導入した山登り法の流れ**\n
    0. パラメータの調整
    1. 通常の山登り(3方向)
    2. 改善が止まる(10ステップ or 20ステップ)
    3. 部分ランダムジャンプ発動（10ステップ：小ジャンプ, 20ステップ：大ジャンプ）
      - 小ジャンプ強度：[0.85,1.15](±15%), 大ジャンプ強度：[0.5,1.5](±50%)のノイズをかける
      - ジャンプさせる特徴量：ランダム(小ジャンプ：特徴量数の10%を選択,大ジャンプ：特徴量数の20%を選択)
    4. クリップ・正規化で整える
    5. 新しい重みでSVM再学習・スコア評価
    6. スコアが良化すれば採用
    7. 1〜6をグリッドサーチのパラメータ分だけ繰り返す\n
    **正規化を行う理由**\n
    - 山登り法がムダな探索をしないようにするため.
      - 例：A[1.0, 1.5, 0.5](平均 1.0),B[2.0, 3.0, 1.0](平均 2.0),C[0.8, 1.2, 0.4](平均 0.8)の３つの重みがあるとする.
      これらA,B,Cは、重み間の比率は全く同じであるため、SVMで評価すると、これらは非常に似た（あるいは全く同じ）スコアになり、重みの比率が全く同じで、全体のスケール（大きさ）だけが違う解を、延々と追いかけてしまう可能性がある.
    - SVMに見えるスケールを一定に保つため.
      - 特徴量xを2倍すると境界のバランスをとるために1/2するように特徴量のスケールに合わせてスケーリングする.すると、ハイパーパラメータCも変化する.
      - RBFカーネルの場合、スケールの影響を強く受ける.入力データが2倍になると距離が2^2倍になり、指数関数的に劇的に変化させてしまう.
    """) 

    st.latex(r"""
    \begin{align*}
    \min_{w, b, \xi} \quad & \frac{1}{2} \|w\|^2 + C \sum_{i \in \{1 \ldots n\}} \xi_i \\
    \text{s.t.} \quad & y_i(w^\top x_i + b) \geq 1 - \xi_i \quad (i \in \{1 \ldots n\}) \\
    & \xi_i \geq 0
    \end{align*}
    """)

    st.latex(r"K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)")

    st.markdown("""
    **実験概要**
      - 質問表：PainDETECT(質問項目:13)，BS-POP(質問項目:18)，及び両者を結合したFUSION(質問項目:31)
      - 特徴量エンジニアリング：PCA、欠損値削除、平均 0・分散 1に標準化
      - 分類モデル：SVM
      - カーネル：線形カーネル，多項式カーネル
      - 性能評価：5分割交差検証
      - パラメータチューニング（C,γ,d,c）：グリッドサーチ
        - 線形カーネル：C(0.1,1) 2通り
        - 多項式カーネル：C(0.1,1),gamma(0.01),degree(2,3),coef0(3,5,8) 12通り
      - 各特徴量に対して重み付け (x′ = α × x )：山登り法（初期値：1，ステップ幅：0.01，反復回数：1000)
      - 主成分数(80%)： 7次元(PainDETECT)、11次元(BS-POP)、18次元(FUSION)\n
    **結果・考察**
      - ランダムジャンプを入れることで以前より少しスコアを向上させることができた.
      - 一番計算量の多い多項式カーネル(12通りのグリッドサーチ × 山登り回数1000回)で2746秒(約46分)かかる.
      - 各質問表・カールごとに１回しか実験できていないため、各重みの重要度は測れない.
    """)

    img_1 = Image.open('picture/20251107/randomjamp_score.png')
    st.image(img_1)

    csv_file_path = 'picture/20251107/randomjamp_weights.csv'  # ファイルパスを指定
    df = pd.read_csv(csv_file_path)
    st.dataframe(df)

    # タブの作成
    tab1, tab2, tab3 = st.tabs(["PainDETECT", "BS-POP", "FUSION"])
    # 各タブに内容を追加
    with tab1:
      img1 = Image.open('picture/20251107/paindetect_linear.png')
      st.image(img1, caption="[3]線形カーネル", width="stretch")
      img2 = Image.open('picture/20251107/paindetect_poly.png')
      st.image(img2, caption="[4]多項式カーネル", width="stretch")

    with tab2:
      img3 = Image.open('picture/20251107/bspop_linear.png')
      st.image(img3, caption="[5]線形カーネル", width="stretch")
      img4 = Image.open('picture/20251107/bspop_poly.png')
      st.image(img4, caption="[6]多項式カーネル", width="stretch")

    with tab3:
      img5 = Image.open('picture/20251107/fusion_linear.png')
      st.image(img5, caption="[1]線形カーネル", width="stretch")
      img6 = Image.open('picture/20251107/fusion_poly.png')
      st.image(img6, caption="[2]多項式カーネル", width="stretch")



with st.container(border=True):
    st.subheader("今後の予定の確認", divider='rainbow')
    st.markdown("""
    - 神経障害性疼痛に焦点を当てた目的関数で最適化を行う
      - OVR（One-Versus-Rest)：各クラスに対して「そのクラスとその他全てのクラス」の分類器を作成する
      - 特徴量重要度を測ることができるPermutation Importanceがある
    - ランダムジャンプを導入した山登り法をガウスカーネル・Sigmoidカーネルでも実験する
    - 他の欠損値補完での実験
    - パラメータの探索範囲を細かくする
    - 遺伝的アルゴリズムによる重み付け
    - 内積=類似度、コサイン類似度を調べる
    - シグモイドカーネルはonとoffを調べている
    """)