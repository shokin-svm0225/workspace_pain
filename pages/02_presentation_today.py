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
    - 中間発表予稿の内容
    - 主成分分析
    - 今後の予定の確認
    - アドバイス
    """)

with st.container(border=True):
    st.subheader("中間発表予稿の内容", divider='rainbow')
    st.markdown("""
    - 実験概要
      - 質問表：PainDETECT(質問項目:13)，BS-POP(質問項目:18)，及び両者を結合したFUSION(質問項目:31)
      - PainDETECTスコア総和による判定：正答率は55.4%，感度/特異度は侵害受容性疼痛 80.1%/38.8%，神経障害性疼痛 15.3%/94.2%，原因不明 21.9%/81.9% \n
      参考文献：https://mhlw-grants.niph.go.jp/system/files/2013/133141/201323002A/201323002A0011.pdf
      - 特徴量エンジニアリング：欠損値削除、平均 0・分散 1に標準化
      - 分類モデル：SVM
      - カーネル：線形カーネル，多項式カーネル，ガウスカーネル，シグモイドカーネル
      - 性能評価：5分割交差検証
      - パラメータチューニング（C,γ,d,c）：グリッドサーチ（探索回数：1000回）
      - 各特徴量に対して重み付け (x′ = α × x )：山登り法（初期値：1，ステップ幅：0.01，反復回数：1000).
    - 結果・考察
      - 機械学習モデルとして SVM を用いることで，PainDETECT のスコア総和による判定より高精度になることがわかる．特に，神経障害性疼痛の感度が約15%から約50%近く向上させることができた．
      - 標準SVMに対して質問項目別の重み付けを導入し山登り法で最適化することで，小幅ながら一貫した精度向上が確認できた．
      - クラス別の結果を見ると，侵害受容性疼痛の感度は高いが，神経障害性疼痛の感度が相対的に低いことから，これが正答率に影響している可能性がある．
      - 必ずしもこの特徴量にこの重み付けが適切だというものはない
    """)
    img4 = Image.open('picture/20251017/hillcliming_score.png')
    st.image(img4, caption='表1', width="stretch")
    st.markdown("""
    - 表2
      - 初期値：①全て1(Linearだけ-step_size：0.1、その他-step_size：0.01)、②ランダム(-5〜5)(step_size：0.01)、③全て1(step_size：0.01)
    """)
   # タブの作成
    tab1, tab2, tab3 = st.tabs(["PainDETECT", "BS-POP", "FUSION"])
    # 各タブに内容を追加
    with tab1:
      img1 = Image.open('picture/20251017/20251017_PAINDETECT.png')
      st.image(img1, width="stretch")

    with tab2:
      img2 = Image.open('picture/20251017/20251017_BSPOP.png')
      st.image(img2, width="stretch")

    with tab3:
      img3 = Image.open('picture/20251017/20251017_FUSION.png')
      st.image(img3, width="stretch")

        
with st.container(border=True):
    st.subheader("主成分分析", divider='rainbow')
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
        - 多次元の質問項目間に存在する相関を明らかにする
        - データのグラフ化
        - 計算時間の短縮\n
        **検証** \n
        3つの質問表に対して主成分分析をし、主成分の特徴や分離度を確認した.
        - 特徴量エンジニアリング：欠損値削除,標準化
        - 質問表：PainDETECT(質問項目:13)，BS-POP(質問項目:18)，及び両者を結合したFUSION(質問項目:31)\n
        **結果・考察**
        - PainDETECTでは、P1-P3とP4-P5、P7-P13に相関関係がありそう
        - BS-POPでは、P1-P3とD11-D18に相関関係がありそう
        - 主成分の上位寄与項目
          - PainDETECT(PC1)：P3, P1, P2, P8, P12
          - Bs-POP(PC1)：D3, D2, D4, D1, D7
          - FUSION(PC1)：P3, P1, D2, P8, P2
        - 累積寄与率(90%)： 9次元(PainDETECT)、14次元(PainDETECT)、22次元(FUSION)
        - PainDETECTでは、PC1の寄与率が高い、PC1〜PC2で十分可視化できる
        - 各主成分は複数の質問項目の混合となるため、直接この質問項目が影響しているとは言えない \n 
        **実験** \n 
        上記に記載している実験概要の内容で、PCAを導入して精度を測る 
        - 線形カーネルのみ
        - 主成分数(90%)：9次元(PainDETECT)、14次元(BS-POP)、22次元(FUSION)\n 
        **実験結果** 
        """)
    st.markdown("""
    - PainDETECTでは、PCA前より約1%向上した
    - ３つの質問表とも第１主成分が重み付けの値が高く出るわけではなく、目的関数を目視的に分類できなかった
    """)
   # タブの作成
    tab1, tab2, tab3 = st.tabs(["PainDETECT", "BS-POP", "FUSION"])
    # 各タブに内容を追加
    with tab1:
      img1 = Image.open('picture/20251017/pca_painDETECT1.png')
      st.image(img1, width="stretch")
      img2 = Image.open('picture/20251017/pca_painDETECT2.png')
      st.image(img2, width="stretch")

    with tab2:
      img4 = Image.open('picture/20251017/pca_bspop1.png')
      st.image(img4, width="stretch")
      img5 = Image.open('picture/20251017/pca_bspop2.png')
      st.image(img5, width="stretch")
      img6 = Image.open('picture/20251017/pca_bspop3.png')
      st.image(img6, width="stretch")

    with tab3:
      img7 = Image.open('picture/20251017/pca_fusion1.png')
      st.image(img7, width="stretch")
      img8 = Image.open('picture/20251017/pca_fusion2.png')
      st.image(img8, width="stretch")
      img9 = Image.open('picture/20251017/pca_fusion3.png')
      st.image(img9, width="stretch")


with st.container(border=True):
    st.subheader("今後の予定の確認", divider='rainbow')
    st.markdown("""
    - プログラム中の並行処理→並列処理へ
    - 神経障害性疼痛に焦点を当てた目的関数で最適化を行う
      - OVR（One-Versus-Rest)：各クラスに対して「そのクラスとその他全てのクラス」の分類器を作成する
      - 特徴量重要度を測ることができるPermutation Importanceがある
    - 他の欠損値補完での実験
    - パラメータの探索範囲を細かくする
    - 遺伝的アルゴリズムによる重み付け
    """)