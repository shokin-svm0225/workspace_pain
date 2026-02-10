import streamlit as st
import itertools
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from streamlit_option_menu import option_menu
ImageFile.LOAD_TRUNCATED_IMAGES = True

st.title('実験結果の報告')

with st.container(border=True):
    st.subheader("目次", divider='rainbow')
    st.markdown("""
    - データとカーネル関数の定式に基づくハイパーパラメータの検討
      - BS-POPの結果の特性を元に自身で値を定めて実験
      - その他の質問表・カーネルの実験結果
      - 主成分分析を行ったデータでのカーネル関数の定式に基づくハイパーパラメータの検討と実験
    - SOMによる検証
    - 今後の予定の確認
    - アドバイス
    """)

with st.container(border=True):
    st.subheader("データとカーネル関数の定式に基づくハイパーパラメータの検討", divider='rainbow')
    st.subheader("標準化あり・なし")
    st.markdown("""
    SVMにおけるカーネル関数の定式に着目し、同一の診断結果を持つデータ同士と異なる診断結果を持つデータ同士をそれぞれ抽出し、
    それらに対して行列計算（内積・ユーグリッド距離）を自身で行うことで、カーネル関数における最適なハイパーパラメータの候補を調べる。\n
    特に、ハイパーパラメータの調整が必要な多項式カーネル、ガウス（RBF）カーネル、シグモイドカーネルについて、
    定式に基づく計算結果からハイパーパラメータの妥当な範囲について考察する。
                
    **カーネル関数**\n
    - 線形カーネル
    """)
    st.latex(r"K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{x}'")
    st.markdown("""
    - 多項式カーネル
    """)
    st.latex(r"K(\mathbf{x}, \mathbf{x}') = (\gamma \mathbf{x}^\top \mathbf{x}' + r)^d")
    st.markdown("パラメータ: $d$ (degree), $\gamma$ (gamma), $r$ (coef0)")
    st.markdown("""
    - ガウス（RBF）カーネル
    """)
    st.latex(r"K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)")
    st.markdown("パラメータ: $\gamma$ (gamma)")
    st.markdown("""
    - シグモイドカーネル
    """)
    st.latex(r"K(\mathbf{x}, \mathbf{x}') = \tanh(\gamma \mathbf{x}^\top \mathbf{x}' + r)")
    st.markdown("パラメータ: $\gamma$ (gamma), $r$ (coef0)")

    st.markdown("""
    **検証**\n
    前回SVMにおけるカーネル関数の定式に着目し、同一の診断結果を持つデータ同士と異なる診断結果を持つデータ同士をそれぞれ抽出し、
    それらに対して行列計算（内積・ユーグリッド距離）を自身で行った。その中で、BS-POPの内積の結果に同一の診断結果を持つデータ同士と異なる診断結果を持つデータ同士で
    正負に分離される特性が見られたので、実験を行い精度を図る。
    - 疼痛の種類ごとのデータ数
      - 侵害受容性疼痛：331
      - 神経障害性疼痛：177
      - 不明：32
    - 標準化の有無：平均0、分散1に変換
    - 使用する質問表
      - BS-POP（18項目）
    - 結果の出力
      - 各組み合わせに対して全データペアについて内積を計算し、その平均値を表示
      - 同一疼痛同士の比較では、自分自身とのペアを除外して平均を計算する
      - 内積・ユークリッド距離の平均における標準偏差も出力し、ばらつきを確認する\n
    **検証結果**\n
    これまでの実験の結果より精度が明確に上がることはみ見られなかった。\n
    :red[内積・ユークリッド距離の平均における標準偏差を出力し、ばらつきを確認したところ[上記に検証結果記載]、
    標準化ありで約5-6の間に収束しており、ばらつきがあるため必ずしも正負に分けられないことから精度が上がらないと考察できる。]
    """)
    # タブの作成
    tab1, tab2 = st.tabs(["標準化あり", "標準化なし"])
    # 各タブに内容を追加
    with tab1:
        st.markdown("- PainDITECT")
        csv_file_path_1 = 'picture/20260116/標準化有_painditect.csv'  # ファイルパスを指定
        df1 = pd.read_csv(csv_file_path_1)
        st.dataframe(df1)
        st.markdown("- BS-POP")
        csv_file_path_2 = 'picture/20260116/標準化有_bspop.csv'  # ファイルパスを指定
        df2 = pd.read_csv(csv_file_path_2)
        st.dataframe(df2)
        st.markdown("- FUSION")
        csv_file_path_3 = 'picture/20260116/標準化有_fusion.csv'  # ファイルパスを指定
        df3 = pd.read_csv(csv_file_path_3)
        st.dataframe(df3)

    with tab2:
        st.markdown("- PainDITECT")
        csv_file_path_1 = 'picture/20260116/標準化無_painditect.csv'  # ファイルパスを指定
        df1 = pd.read_csv(csv_file_path_1)
        st.dataframe(df1)
        st.markdown("- BS-POP")
        csv_file_path_2 = 'picture/20260116/標準化無_bspop.csv'  # ファイルパスを指定
        df2 = pd.read_csv(csv_file_path_2)
        st.dataframe(df2)
        st.markdown("- FUSION")
        csv_file_path_3 = 'picture/20260116/標準化無_fusion.csv'  # ファイルパスを指定
        df3 = pd.read_csv(csv_file_path_3)
        st.dataframe(df3)

    st.markdown("""
    **実験1**\n
    検証にて求めたデータにおける内積・ユークリッド距離から考察する各カーネルの最適なハイパーパラメータの候補を元に、山登り法＋ランダムジャンプを用いたSVMでの実装を行う。
    - 山登り法(2方向)の反復回数：1000
    - Step Size：0.01 ※加算
    - ランダムジャンプ
      - スコアの停滞が10回で小ジャンプ、20回で大ジャンプ
      - 全特徴量（PCAの数）のうち、小ジャンプ(10%) / 大ジャンプ(20%)の重みを強制的に変更
      - 強度：小ジャンプ[0.85,1.15](±15%) / 大ジャンプ[0.5,1.5](±50%)
    - 標準化の有無：平均0、分散1に変換
    - 使用する質問表
      - PainDETECT（13項目）
      - BS-POP（18項目）
      - FUSION（31項目）
    - グリッドサーチによるパラメータチューニング
      - C：0.1,1
      - gamma：検証で定めた候補を用いる
      - coef0：0
      - degree：2,3
    - 結果の出力
      - 最もスコアの高いモデル（スコア・重み・パラメータ）を出力
      - 山登り法によるスコアの変化を可視化
    - :red[グリッドサーチ値の候補の定め方]
      - :red[ガウスカーネル：ユークリッドの距離の二乗の平均 × ◼︎が1/e（0.37）になるような◼︎を求めて、その前後を候補にする]
      - :red[シグモイドカーネル：内積の平均 × ◼︎ がtanh（1,-1）の間に収まるような◼︎を候補にする。また、coef0はgammaで抑えられるので0とする。]
      - :red[多項式カーネル：内積の平均 × ◼︎ が1付近に収まるような◼︎を候補にする。また、coef0はgammaで抑えられるので0とし、degreeは大きく値が変化するのを防ぐため2,3とする。]
    """)

    st.markdown("""
    **実験結果**\n
    内積、ユークリッド距離ともに同じ診断結果のデータ同士・異なる診断結果のデータ同士での明確な違いが見られなかった。
    しかし、前回まではgamma：3,5,8と定めていたが、自身で行列計算すると、異なる範囲を探索していることがわかった。
    - 標準化なし
      - ガウスカーネル（gamma）\n
        - 予測（BS-POP）：ユークリッドの距離のおよそ平均で10とすると、1/e（0.37）を基準で、[0.1]\n
          実験のグリッドサーチ値：:red[gamma：0.01,0.1,0.2]\n
          実行時間：3310秒\n
          最高スコア：63.89%
    """)
    st.dataframe(pd.read_csv('picture/20260210/rbf/score_rbf_bspop_no.csv'))
    st.markdown(""" 
      - シグモイドカーネル（gamma,coef0）\n
        - 予測（BS-POP）：内積のおよそ平均で40とすると、tanh（1,-1）を基準で、[gamma：-0.025〜0.025]、gammaで抑えられるので[coef0：0]\n
          実験のグリッドサーチ値：:red[gamma：0.01,0.02,0.025], :red[coef0：0]\n
          実行時間：2278秒\n
          最高スコア：61.67%
    """)
    st.dataframe(pd.read_csv('picture/20260210/sigmoid/score_sigmoid_bspop_no.csv'))
    st.markdown("""
      - 多項式カーネル（gamma,coef0,degree）\n
        - 予測（BS-POP）：内積のおよそ平均で40とすると、[gamma：0.025付近]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
          実験のグリッドサーチ値：:red[gamma：30,80,100], :red[coef0：0], :red[degree：2,3]\n
          実行時間：3479秒\n
          最高スコア：64.07%          
    """)
    st.dataframe(pd.read_csv('picture/20260210/poly/score_poly_bspop_no.csv'))

    st.markdown("""
    - 標準化あり
      - ガウスカーネル（gamma）\n
        - 予測（BS-POP）：ユークリッドの距離のおよそ平均で37とすると、1/e（0.37）を基準で、[0.027]\n
          実験のグリッドサーチ値：:red[gamma：0.01,0.027,0.1]\n
          実行時間：3284秒\n
          最高スコア：65.74%
    """)
    st.dataframe(pd.read_csv('picture/20260210/rbf/score_rbf_bspop_yes.csv'))
    st.markdown("""
      - シグモイドカーネル（gamma,coef0）\n 
        - 予測（BS-POP）：内積のおよそ平均で0.002とすると、tanh（1,-1）を基準で、[gamma：-500〜500]、gammaで抑えられるので[coef0：0]\n
          実験のグリッドサーチ値：:red[gamma：50,250,500], :red[coef0：0]\n
          実行時間：1747秒\n
          最高スコア：62.96%
    """)
    st.dataframe(pd.read_csv('picture/20260210/sigmoid/score_sigmoid_bspop_yes.csv'))
    st.markdown("""
      - 多項式カーネル（gamma,coef0,degree）\n
        - 予測（BS-POP）：内積のおよそ平均で0.002とすると、[gamma：500付近]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
          実験のグリッドサーチ値：:red[gamma：400,500,600], :red[coef0：0], :red[degree：2,3]\n
          実行時間：まだ実装中 秒\n
          最高スコア：- %          
    """)
    # st.dataframe(pd.read_csv('picture/20260210/poly/score_poly_bspop_yes.csv'))

    st.markdown("""
    **実験2**\n
    検証にて求めたデータにおける内積・ユークリッド距離から考察する各カーネルの最適なハイパーパラメータの候補を元に、山登り法＋ランダムジャンプを用いたSVMでの実装をBS-POP以外でも行う。
    - 山登り法(2方向)の反復回数：1000
    - Step Size：0.01 ※加算
    - ランダムジャンプ
      - スコアの停滞が10回で小ジャンプ、20回で大ジャンプ
      - 全特徴量（PCAの数）のうち、小ジャンプ(10%) / 大ジャンプ(20%)の重みを強制的に変更
      - 強度：小ジャンプ[0.85,1.15](±15%) / 大ジャンプ[0.5,1.5](±50%)
    - 標準化の有無：平均0、分散1に変換
    - 使用する質問表
      - PainDETECT（13項目）
      - BS-POP（18項目）
      - FUSION（31項目）
    - グリッドサーチによるパラメータチューニング
      - C：0.1,1
      - gamma：検証で定めた候補を用いる
      - coef0：0
      - degree：2,3
    - 結果の出力
      - 最もスコアの高いモデル（スコア・重み・パラメータ）を出力
      - 山登り法によるスコアの変化を可視化
    - :red[グリッドサーチ値の候補の定め方]
      - :red[ガウスカーネル：ユークリッドの距離の二乗の平均 × ◼︎が1/e（0.37）になるような◼︎を求めて、その前後を候補にする]
      - :red[シグモイドカーネル：内積の平均 × ◼︎ がtanh（1,-1）の間に収まるような◼︎を候補にする。また、coef0はgammaで抑えられるので0とする。]
      - :red[多項式カーネル：内積の平均 × ◼︎ が1付近に収まるような◼︎を候補にする。また、coef0はgammaで抑えられるので0とし、degreeは大きく値が変化するのを防ぐため2,3とする。]
    """)

    st.markdown("""
    **実験結果**\n
    内積、ユークリッド距離ともに同じ診断結果のデータ同士・異なる診断結果のデータ同士での明確な違いが見られなかった。
    しかし、前回まではgamma：3,5,8と定めていたが、自身で行列計算すると、異なる範囲を探索していることがわかった。
    - 標準化なし
      - ガウスカーネル（gamma）\n
        - 予測（PainDETECT）：ユークリッドの距離のおよそ平均で60とすると、1/e（0.37）を基準で、[0.016]\n
          実験のグリッドサーチ値：:red[gamma：0.01,0.016,0.05]\n
          実行時間：2191秒\n
          最高スコア：72.04%
    """)
    st.dataframe(pd.read_csv('picture/20260210/rbf/score_rbf_paindetect_no.csv'))
    st.markdown("""
        - 予測（BS-POP）：ユークリッドの距離のおよそ平均で10とすると、1/e（0.37）を基準で、[0.1]\n
          実験のグリッドサーチ値：:red[gamma：0.01,0.1,0.2]\n
          実行時間：3310秒\n
          最高スコア：63.89%
    """)
    st.dataframe(pd.read_csv('picture/20260210/rbf/score_rbf_bspop_no.csv'))
    st.markdown("""
        - 予測（FUSION）：ユークリッドの距離のおよそ平均で70とすると、1/e（0.37）を基準で、[0.014]\n
          実験のグリッドサーチ値：:red[gamma：0.01,0.014,0.03]\n
          実行時間：6869秒\n
          最高スコア：72.59%
    """)
    st.dataframe(pd.read_csv('picture/20260210/rbf/score_rbf_fusion_no.csv'))
    st.markdown("""   
      - シグモイドカーネル（gamma,coef0）\n
        - 予測（PainDETECT）：内積のおよそ平均で130とすると、tanh（1,-1）を基準で、[gamma：-0.008〜0.008]、gammaで抑えられるので[coef0：0]\n
          実験のグリッドサーチ値：:red[gamma：0.001,0.004,0.008], :red[coef0：0]\n
          実行時間：1517秒\n
          最高スコア：71.67%
    """)
    st.dataframe(pd.read_csv('picture/20260210/sigmoid/score_sigmoid_paindetect_no.csv'))
    st.markdown(""" 
        - 予測（BS-POP）：内積のおよそ平均で40とすると、tanh（1,-1）を基準で、[gamma：-0.025〜0.025]、gammaで抑えられるので[coef0：0]\n
          実験のグリッドサーチ値：:red[gamma：0.01,0.02,0.025], :red[coef0：0]\n
          実行時間：2278秒\n
          最高スコア：61.67%
    """)
    st.dataframe(pd.read_csv('picture/20260210/sigmoid/score_sigmoid_bspop_no.csv'))
    st.markdown(""" 
        - 予測（FUSION）：内積のおよそ平均で170とすると、tanh（1,-1）を基準で、[gamma：-0.006〜0.006]、gammaで抑えられるので[coef0：0]\n
          実験のグリッドサーチ値：:red[gamma：0.001,0.003,0.006], :red[coef0：0]\n
          実行時間：4732秒\n
          最高スコア：67.22%
    """)
    st.dataframe(pd.read_csv('picture/20260210/sigmoid/score_sigmoid_fusion_no.csv'))
    st.markdown(""" 
      - 多項式カーネル（gamma,coef0,degree）\n
        - 予測（PainDETECT）：内積のおよそ平均で130とすると、[gamma：0.008付近]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
          実験のグリッドサーチ値：:red[gamma：0.003,0.008,0.01], :red[coef0：0], :red[degree：2,3]\n
          実行時間：2813秒\n
          最高スコア：72.04%
    """)
    st.dataframe(pd.read_csv('picture/20260210/poly/score_poly_paindetect_no.csv'))
    st.markdown("""
        - 予測（BS-POP）：内積のおよそ平均で40とすると、[gamma：0.025付近]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
          実験のグリッドサーチ値：:red[gamma：30,80,100], :red[coef0：0], :red[degree：2,3]\n
          実行時間：3479秒\n
          最高スコア：64.07%          
    """)
    st.dataframe(pd.read_csv('picture/20260210/poly/score_poly_bspop_no.csv'))
    st.markdown("""
        - 予測（FUSION）：内積のおよそ平均で170とすると、[gamma：0.006付近]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
          実験のグリッドサーチ値：:red[gamma：0.001,0.006,0.01], :red[coef0：0], :red[degree：2,3]\n
          実行時間：7976秒\n
          最高スコア：72.41%
    """)
    st.dataframe(pd.read_csv('picture/20260210/poly/score_poly_fusion_no.csv'))

    st.markdown("""
    - 標準化あり
      - ガウスカーネル（gamma）\n
        - 予測（PainDETECT）：ユークリッドの距離のおよそ平均で25とすると、1/e（0.37）を基準で、[0.04]\n
          実験のグリッドサーチ値：:red[gamma：0.01,0.04,0.1]\n
          実行時間：2159秒\n
          最高スコア：72.41%
    """)
    st.dataframe(pd.read_csv('picture/20260210/rbf/score_rbf_paindetect_yes.csv'))
    st.markdown("""
        - 予測（BS-POP）：ユークリッドの距離のおよそ平均で37とすると、1/e（0.37）を基準で、[0.027]\n
          実験のグリッドサーチ値：:red[gamma：0.01,0.027,0.1]\n
          実行時間：3284秒\n
          最高スコア：65.74%
    """)
    st.dataframe(pd.read_csv('picture/20260210/rbf/score_rbf_bspop_yes.csv'))
    st.markdown("""
        - 予測（FUSION）：ユークリッドの距離のおよそ平均で63とすると、1/e（0.37）を基準で、[0.016]\n
          実験のグリッドサーチ値：:red[gamma：0.01,0.016,0.04]\n
          実行時間：6983秒\n
          最高スコア：72.96%
    """)
    st.dataframe(pd.read_csv('picture/20260210/rbf/score_rbf_fusion_yes.csv'))
    st.markdown("""   
      - シグモイドカーネル（gamma,coef0）\n
        - 予測（PainDETECT）：内積のおよそ平均で0.18とすると、tanh（1,-1）を基準で、[gamma：-5.555〜5.555]、gammaで抑えられるので[coef0：0]\n
          実験のグリッドサーチ値：:red[gamma：1,3,5.5], :red[coef0：0]\n
          実行時間：1782秒\n
          最高スコア：65.37%
    """)
    st.dataframe(pd.read_csv('picture/20260210/sigmoid/score_sigmoid_paindetect_yes.csv'))
    st.markdown(""" 
        - 予測（BS-POP）：内積のおよそ平均で0.002とすると、tanh（1,-1）を基準で、[gamma：-500〜500]、gammaで抑えられるので[coef0：0]\n
          実験のグリッドサーチ値：:red[gamma：50,250,500], :red[coef0：0]\n
          実行時間：1747秒\n
          最高スコア：62.96%
    """)
    st.dataframe(pd.read_csv('picture/20260210/sigmoid/score_sigmoid_bspop_yes.csv'))
    st.markdown(""" 
        - 予測（FUSION）：内積のおよそ平均で0.18とすると、tanh（1,-1）を基準で、[gamma：-5.555〜5.555]、gammaで抑えられるので[coef0：0]\n
          実験のグリッドサーチ値：:red[gamma：1,3,5.5], :red[coef0：0]\n
          実行時間：4545秒\n
          最高スコア：73.15%
    """)
    st.dataframe(pd.read_csv('picture/20260210/sigmoid/score_sigmoid_fusion_yes.csv'))
    st.markdown(""" 
      - 多項式カーネル（gamma,coef0,degree）\n
        - 予測（PainDETECT）：内積のおよそ平均で0.18とすると、[gamma：5.555付近]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
          実験のグリッドサーチ値：:red[gamma：5,5.5,6], :red[coef0：0], :red[degree：2,3]\n
          実行時間：4047秒\n
          最高スコア：61.85%
    """)
    st.dataframe(pd.read_csv('picture/20260210/poly/score_poly_paindetect_yes.csv'))
    st.markdown("""
        - 予測（BS-POP）：内積のおよそ平均で0.002とすると、[gamma：500付近]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
          実験のグリッドサーチ値：:red[gamma：400,500,600], :red[coef0：0], :red[degree：2,3]\n
          実行時間：まだ実装中 秒\n
          最高スコア：- %          
    """)
    # st.dataframe(pd.read_csv('picture/20260210/poly/score_poly_bspop_yes.csv'))
    st.markdown("""
        - 予測（FUSION）：内積のおよそ平均で0.18とすると、[gamma：5.555付近]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
          実験のグリッドサーチ値：:red[gamma：5,5.5,6], :red[coef0：0], :red[degree：2,3]\n
          実行時間：まだ実装中 秒\n
          最高スコア：- %
    """)
    # st.dataframe(pd.read_csv('picture/20260210/poly/score_poly_fusion_yes.csv'))

with st.container(border=True):
    st.subheader("データとカーネル関数の定式に基づくハイパーパラメータの検討", divider='rainbow')
    st.subheader("主成分分析")
    st.markdown("""
    上記内容と同様
    """)
    st.markdown("""
    **検証**\n
    - 疼痛の種類ごとのデータ数
      - 侵害受容性疼痛：331
      - 神経障害性疼痛：177
      - 不明：32
    - 標準化：平均0、分散1に変換
    - 使用する質問表(主成分分析後の使用する特徴量数：寄与率80%)
      - PainDETECT（13項目 → 7項目）
      - BS-POP（18項目 → 11項目）
      - FUSION（31項目 → 18項目）
    - 結果の出力
      - 各組み合わせに対して全データペアについて内積を計算し、その平均値を表示
      - 同一疼痛同士の比較では、自分自身とのペアを除外して平均を計算する
      - 内積・ユークリッド距離の平均における標準偏差も出力し、ばらつきを確認する\n
    **検証結果**\n
    内積、ユークリッド距離ともに同じ診断結果のデータ同士・異なる診断結果のデータ同士での明確な違いが見られなかった。
    BS-POPに関しては、内積が同じ診断結果のデータ同士・異なる診断結果のデータ同士で正負で分けられているように見えるが、標準偏差を見ると約5-6でばらつきがあるため必ずしも正負に分けれなそう。
    """)
    st.markdown("- PainDITECT")
    csv_file_path_1 = 'picture/20260210/pca_painditect.csv'  # ファイルパスを指定
    df1 = pd.read_csv(csv_file_path_1)
    st.dataframe(df1)
    st.markdown("- BS-POP")
    csv_file_path_2 = 'picture/20260210/pca_bspop.csv'  # ファイルパスを指定
    df2 = pd.read_csv(csv_file_path_2)
    st.dataframe(df2)
    st.markdown("- FUSION")
    csv_file_path_3 = 'picture/20260210/pca_fusion.csv'  # ファイルパスを指定
    df3 = pd.read_csv(csv_file_path_3)
    st.dataframe(df3)

    st.markdown("""
    **実験**\n
    検証にて求めたデータにおける内積・ユークリッド距離から考察する各カーネルの最適なハイパーパラメータの候補を元に、山登り法＋ランダムジャンプを用いたSVMでの実装を行う。
    - 山登り法(2方向)の反復回数：1000
    - Step Size：0.01 ※加算
    - ランダムジャンプ
      - スコアの停滞が10回で小ジャンプ、20回で大ジャンプ
      - 全特徴量（PCAの数）のうち、小ジャンプ(10%) / 大ジャンプ(20%)の重みを強制的に変更
      - 強度：小ジャンプ[0.85,1.15](±15%) / 大ジャンプ[0.5,1.5](±50%)
    - 標準化の有無：平均0、分散1に変換
    - 使用する質問表(主成分分析後の使用する特徴量数：寄与率80%)
      - PainDETECT（13項目 → 7項目）
      - BS-POP（18項目 → 11項目）
      - FUSION（31項目 → 18項目）
    - グリッドサーチによるパラメータチューニング
      - C：0.1,1
      - gamma：検証で定めた候補を用いる
      - coef0：0
      - degree：2,3
    - 結果の出力
      - 最もスコアの高いモデル（スコア・重み・パラメータ）を出力
      - 山登り法によるスコアの変化を可視化
    """)

    st.markdown("""
    **実験結果**\n
    内積、ユークリッド距離ともに同じ診断結果のデータ同士・異なる診断結果のデータ同士での明確な違いが見られなかった。
    しかし、前回まではgamma：3,5,8と定めていたが、自身で行列計算すると、異なる範囲を探索していることがわかった。
    - 主成分分析
      - ガウスカーネル（gamma）\n
        - 予測（PainDETECT）：ユークリッドの距離のおよそ平均で20とすると、1/e（0.37）を基準で、[0.05]\n
          実験のグリッドサーチ値：:red[gamma：0.02,0.05,0.08]\n
          実行時間：996秒\n
          最高スコア：72.59%
    """)
    st.dataframe(pd.read_csv('picture/20260210/rbf/score_rbf_paindetect_pca.csv'))
    st.markdown("""
        - 予測（BS-POP）：ユークリッドの距離のおよそ平均で30とすると、1/e（0.37）を基準で、[0.03]\n
          実験のグリッドサーチ値：:red[gamma：0.01,0.03,0.05]\n
          実行時間：1884秒\n
          最高スコア：63.15%
    """)
    st.dataframe(pd.read_csv('picture/20260210/rbf/score_rbf_bspop_pca.csv'))
    st.markdown("""
        - 予測（FUSION）：ユークリッドの距離のおよそ平均で52とすると、1/e（0.37）を基準で、[0.02]\n
          実験のグリッドサーチ値：:red[gamma：0.01,0.02,0.1]\n
          実行時間：3464秒\n
          最高スコア：71.85%
    """)
    st.dataframe(pd.read_csv('picture/20260210/rbf/score_rbf_fusion_pca.csv'))
    st.markdown("""   
      - シグモイドカーネル（gamma,coef0）\n
        - 予測（PainDETECT）：内積のおよそ平均で0.16とすると、tanh（1,-1）を基準で、[gamma：-6.25〜6.25]、gammaで抑えられるので[coef0：0]\n
          実験のグリッドサーチ値：:red[gamma：3,5,6.25], :red[coef0：0]\n
          実行時間：693秒\n
          最高スコア：68.33%
    """)
    st.dataframe(pd.read_csv('picture/20260210/sigmoid/score_sigmoid_paindetect_pca.csv'))
    st.markdown(""" 
        - 予測（BS-POP）：内積のおよそ平均で0.0125とすると、tanh（1,-1）を基準で、[gamma：-80〜80]、gammaで抑えられるので[coef0：0]\n
          実験のグリッドサーチ値：:red[gamma：10,40,80], :red[coef0：0]\n
          実行時間：1115秒\n
          最高スコア：63.52%
    """)
    st.dataframe(pd.read_csv('picture/20260210/sigmoid/score_sigmoid_bspop_pca.csv'))
    st.markdown(""" 
        - 予測（FUSION）：内積のおよそ平均で0.18とすると、tanh（1,-1）を基準で、[gamma：-5.55〜5.55]、gammaで抑えられるので[coef0：0]\n
          実験のグリッドサーチ値：:red[gamma：1,3,5.5], :red[coef0：0]\n
          実行時間：2104秒\n
          最高スコア：71.11%
    """)
    st.dataframe(pd.read_csv('picture/20260210/sigmoid/score_sigmoid_fusion_pca.csv'))
    st.markdown(""" 
      - 多項式カーネル（gamma,coef0,degree）\n
        - 予測（PainDETECT）：内積のおよそ平均で0.16とすると、[gamma：6.25付近]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
          実験のグリッドサーチ値：:red[gamma：5,6.25,7], :red[coef0：0], :red[degree：2,3]\n
          実行時間：2112秒\n
          最高スコア：62.59%
    """)
    st.dataframe(pd.read_csv('picture/20260210/poly/score_poly_paindetect_pca.csv'))
    st.markdown("""
        - 予測（BS-POP）：内積のおよそ平均で0.0125とすると、[gamma：80付近]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
          実験のグリッドサーチ値：:red[gamma：30,80,100], :red[coef0：0], :red[degree：2,3]\n
          実行時間：2679秒\n
          最高スコア：56.3%          
    """)
    st.dataframe(pd.read_csv('picture/20260210/poly/score_poly_bspop_pca.csv'))
    st.markdown("""
        - 予測（FUSION）：内積のおよそ平均で0.18とすると、[gamma：5.55付近]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
          実験のグリッドサーチ値：:red[gamma：5,5.5,6], :red[coef0：0], :red[degree：2,3]\n
          実行時間：5431秒\n
          最高スコア：65.37%
    """)
    st.dataframe(pd.read_csv('picture/20260210/poly/score_poly_fusion_pca.csv'))

with st.container(border=True):
    st.subheader("SOM（自己組織化マップ）", divider='rainbow')
    st.markdown("""
    - SOM（自己組織化マップ）の実装
      - 侵害受容性疼痛のデータ、神経障害性疼痛のデータ、不明のデータそれぞれでSOMを走らせて
        神経障害性にどれくらい近いか、侵害受容性にどれくらい近いかという指標を少ない特徴量に要約させてそれを元にSVMを実装するという認識であっていますでしょうか。
    """)

with st.container(border=True):
    st.subheader("これからやること", divider='rainbow')
    st.markdown("""
    - 実装を行い、前回までの実験結果の比較を行う
    - 主成分分析した後のデータに対して行列計算（内積・ユークリッド距離）を自身で行うことで、カーネル関数における最適なハイパーパラメータの候補を調べる
    - 線形以外のカーネルで交互最適化の実験を行う
    - 進化計算での山登り法の実装
    - 他の欠損値補完での実験
    - 教師なし学習の実装
      - som（クラスタリング）
    - 決定木やるならlightgbm
    """)