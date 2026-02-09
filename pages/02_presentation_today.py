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
    - データとカーネル関数の定式に基づくハイパーパラメータの検討
    - 修論タイトルを考える
    - 今後の予定の確認
    - アドバイス
    """)

with st.container(border=True):
    st.subheader("データとカーネル関数の定式に基づくハイパーパラメータの検討", divider='rainbow')
    st.markdown("""
    SVMにおけるカーネル関数の定式に着目し、同一の診断結果を持つデータ同士と異なる診断結果を持つデータ同士をそれぞれ抽出し、
    それらに対して行列計算（内積・ユーグリッド距離）を自身で行うことで、カーネル関数における最適なハイパーパラメータの候補を調べる。\n
    特に、ハイパーパラメータの調整が必要な多項式カーネル、ガウス（RBF）カーネル、シグモイドカーネルについて、
    定式に基づく計算結果からハイパーパラメータの妥当な範囲について考察する。
    
    **設定**\n
    - 疼痛の種類ごとのデータ数
      - 侵害受容性疼痛：331
      - 神経障害性疼痛：177
      - 不明：32
    - 標準化の有無：平均0、分散1に変換
    - 使用する質問表
      - PainDITECT（13項目）
      - BS-POP（18項目）
      - FUSION（31項目）
    - 結果の出力
      - 各組み合わせに対して全データペアについて内積を計算し、その平均値を表示
      - 同一疼痛同士の比較では、自分自身とのペアを除外して平均を計算する
                
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
    **実験結果**\n
    内積、ユーグリッド距離ともに同じ診断結果のデータ同士・異なる診断結果のデータ同士での明確な違いが見られなかった。
    しかし、前回まではgamma：3,5,8と定めていたが、自身で行列計算すると、異なる範囲を探索していることがわかった。
    - :red[標準化なし]
      - ガウスカーネル（gamma）\n
      PainDITECTは、ユーグリッドの距離のおよそ平均で70とすると、1/e（0.37）を基準で、[0.014]\n
      BS-POPは、ユーグリッドの距離のおよそ平均で10とすると、1/e（0.37）を基準で、[0.1]\n
      FUSIONは、ユーグリッドの距離のおよそ平均で70とすると、1/e（0.37）を基準で、[0.014]\n
      - シグモイドカーネル（gamma,coef0）\n
      PainDITECTは、内積のおよそ平均で140とすると、tanh（1,-1）を基準で、[gamma：-0.007〜0.007]、gammaで抑えられるので[coef0：0]\n
      BS-POPは、内積のおよそ平均で40とすると、tanh（1,-1）を基準で、[gamma：-0.025〜0.025]、gammaで抑えられるので[coef0：0]\n
      FUSIONは、内積のおよそ平均で170とすると、tanh（1,-1）を基準で、[gamma：-0.006〜0.006]、gammaで抑えられるので[coef0：0]\n
      - 多項式カーネル（gamma,coef0,degree）\n
      PainDITECTは、内積のおよそ平均で140とすると、tanh（1,-1）を基準で、[gamma：-0.007〜0.007]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
      BS-POPは、内積のおよそ平均で40とすると、tanh（1,-1）を基準で、[gamma：-0.025〜0.025]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
      FUSIONは、内積のおよそ平均で170とすると、tanh（1,-1）を基準で、[gamma：-0.006〜0.006]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
    - :red[標準化あり]
      - ガウスカーネル（gamma）\n
      PainDITECTは、ユーグリッドの距離のおよそ平均で26とすると、1/e（0.37）を基準で、[0.04]\n
      BS-POPは、ユーグリッドの距離のおよそ平均で36とすると、1/e（0.37）を基準で、[0.028]\n
      FUSIONは、ユーグリッドの距離のおよそ平均で63とすると、1/e（0.37）を基準で、[0.016]\n
      - シグモイドカーネル（gamma,coef0）\n
      PainDITECTは、内積のおよそ平均で0.3とすると、tanh（1,-1）を基準で、[gamma：-3.33〜3.33]、gammaで抑えられるので[coef0：0]\n
      BS-POPは、内積のおよそ平均で0.1とすると、tanh（1,-1）を基準で、[gamma：-10〜10]、gammaで抑えられるので[coef0：0]\n
      FUSIONは、内積のおよそ平均で0.5とすると、tanh（1,-1）を基準で、[gamma：-2〜2]、gammaで抑えられるので[coef0：0]\n
      - 多項式カーネル（gamma,coef0,degree）\n
      PainDITECTは、内積のおよそ平均で0.3とすると、tanh（1,-1）を基準で、[gamma：-3.33〜3.33]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
      BS-POPは、内積のおよそ平均で0.1とすると、tanh（1,-1）を基準で、[gamma：-10〜10]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
      FUSIONは、内積のおよそ平均で0.5とすると、tanh（1,-1）を基準で、[gamma：-2〜2]、gammaで抑えられるので[coef0：0]、大きく変化するのを防ぐため[degree：2,3]\n
    - 主成分分析した後のデータに対して行列計算（内積・ユーグリッド距離）を自身で行うことで、カーネル関数における最適なハイパーパラメータの候補を調べる必要がある。
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

with st.container(border=True):
    st.subheader("これからやること", divider='rainbow')
    st.markdown("""
    - 実装を行い、前回までの実験結果の比較を行う
    - 主成分分析した後のデータに対して行列計算（内積・ユーグリッド距離）を自身で行うことで、カーネル関数における最適なハイパーパラメータの候補を調べる
    - 線形以外のカーネルで交互最適化の実験を行う
    - 進化計算での山登り法の実装
    - 他の欠損値補完での実験
    - 教師なし学習の実装
      - som（クラスタリング）
    - 決定木やるならlightgbm
    """)