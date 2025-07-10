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
    - 前回の内容
    - 実験の概要
    - 実験結果・考察
    - 今後の予定の確認
    - アドバイス
    """)  
        
with st.container(border=True):
    st.subheader("実験の概要①", divider='rainbow')
    st.markdown("""
        各質問項目に対して適切な重み付けを山登り法で行う  \n
        - 各ステップで特徴量すべてに対して[-ε,0,+ε]の3方向で評価、重みの更新を行う
          - 初期値：全て1,乱数 \n
        **準備**
        - 質問表：PainDITACT,FUSION（前回：BS-POP）
        - 欠損値の補完：欠損値削除
        - 標準化：あり
        - 重み付けの初期値
          - 全て1.0
          - 全て乱数
            - 山登り法は初期値によって局所最適解を求めてしまう可能性があるから
        - 試行回数：100
        - 重み更新の大きさ：0.01,0.1,1
        - カーネル：線形カーネル
        - 評価：5-分割交差検証(random_state=42)
        - パラメータチューニング(C)：グリッドサーチ（0.01,0.1,1）
        - 結果の出力：正答率（平均）, 感度 , 特異度, スコアの推移のグラフ
        """)

    st.subheader("実験の概要②", divider='rainbow')
    st.markdown("""
        各質問項目に対して適切な重み付けを山登り法で行う  \n
        - 各ステップで特徴量すべてに対して[-ε,0,+ε]の3方向で評価、重みの更新を行う
          - 初期値：全て1,乱数 \n
        **準備**
        - 質問表：PainDITACT,FUSION（前回：BS-POP）
        - 欠損値の補完：欠損値削除
        - 標準化：あり
        - 重み付けの初期値
          - 全て1.0
          - 全て乱数
            - 山登り法は初期値によって局所最適解を求めてしまう可能性があるから
        - 試行回数：100
        - 重み更新の大きさ：0.1,0.2,0.3,0.4,0.5
          - 理由：標準化したデータが大体±1〜2の範囲に収まっており、0.01だと小さすぎると反応しなく、1や10だと大きすぎて各特徴量に偏りができてしまうため。
        - カーネル：線形カーネル
        - 評価：5-分割交差検証(random_state=42)
        - パラメータチューニング(C)：グリッドサーチ（0.01,0.1,1）
        - 結果の出力：正答率（平均）, 感度 , 特異度, スコアの推移のグラフ
        - 疼痛1：侵害受容性疼痛、疼痛2：神経障害性疼痛、疼痛3：不明
        """)
    
    st.subheader("実験の概要③", divider='rainbow')
    st.markdown("""
        線形カーネルだけでなく、色々なカーネル法で実験を行う（非線形分離を可能にする）
        - シグモイドカーネル
        - 多項式カーネル
        - RBFカーネル（Radial Basis Function）
        各質問項目に対して適切な重み付けを山登り法で行う  \n
        - 各ステップで特徴量すべてに対して[-ε,0,+ε]の3方向で評価、重みの更新を行う
        - 初期値：全て1,乱数 \n
        **準備**
        - 質問表：PainDITACT,BS-POP,FUSION
        - 欠損値の補完：欠損値削除
        - 標準化：あり
        - 各質問項目に対して適切な重み付けを山登り法で行う  
          - 各ステップで特徴量すべてに対して[-ε,0,+ε]の3方向で評価、重みの更新を行う
        - 初期値：全て1,乱数 \n
        - 重み付けの初期値
        - 全て1.0
        - 全て乱数
            - 山登り法は初期値によって局所最適解を求めてしまう可能性があるから
        - 試行回数：100
        - 重み更新の大きさ：0.1,0.2,0.3,0.4,0.5
        - 理由：0.01だと小さすぎると反応しなく、1や10だと大きすぎて各特徴量に偏りができてしまうため。
        - カーネル：線形カーネル
        - 評価：5-分割交差検証(random_state=42)
        - パラメータチューニング(C)：グリッドサーチ（0.01,0.1,1）
        - 結果の出力：正答率（平均）, 感度 , 特異度, スコアの推移のグラフ
        """)
    with st.container(border=True):
            st.subheader("シグモイドカーネル", divider='rainbow')
            st.markdown("""
            ニューラルネットワークにおける活性化関数に似た性質を持つカーネル関数
            """)
            st.latex(r"""
            K(x, y) = \tanh(\alpha \cdot \langle x, y \rangle + c)
            """)
            st.markdown("- $\\alpha$: スケーリング係数（内積の強さ調整）")
            st.markdown("- $c$: バイアス項（シフト）")
            st.markdown("- $\\langle x, y \\rangle$: 特徴ベクトルの内積")
            img = Image.open('picture/20250711/tanh.png')
            st.image(img, caption='シグモイドカーネル', use_container_width=True)
    
    with st.container(border=True):
        st.subheader("多項式カーネル", divider='rainbow')
        st.markdown("""
        データ間の関係性を多項式形式で捉えるカーネル関数
        """)
        st.latex(r"""
        K(x, y) = (\alpha \cdot \langle x, y \rangle + c)^d
        """)
        st.markdown("- $\\alpha$: スケーリング係数")
        st.markdown("- $c$: バイアス項（シフト）")
        st.markdown("- $d$: 多項式の次数")
        st.markdown("- $\\langle x, y \\rangle$: 特徴ベクトルの内積")

    with st.container(border=True):
        st.subheader("RBFカーネル（Radial Basis Function）", divider='rainbow')
        st.markdown("""
        データの類似性を指数的に減衰する形式で計算し、非線形なデータを高次元空間で線形的に分離できるようにする関数
        """)
        st.latex(r"""
        K(x, y) = \exp\left(-\gamma \| x - y \|^2 \right)
        """)
        st.markdown("- $\\gamma$: ガウス関数の幅（大きいほど“近い点”に鋭敏）")
        st.markdown("- $\\| x - y \\|^2$: ユークリッド距離の2乗")


with st.container(border=True):
    st.subheader("実験結果①", divider='rainbow')
    st.markdown("""
    - [前回の実験]PainDITECT・FUSIONの質問表
    """)
    img1 = Image.open('picture/20250711/PAIN.png')
    img2 = Image.open('picture/20250711/FUSION.png')
    st.image(img1, caption='PainDITECT', use_container_width=True)
    st.image(img2, caption="FUSION", use_container_width=True)

    st.subheader("実験結果②", divider='rainbow')
    st.markdown("""
    - 更新幅を小さい範囲で山登り法を実施
    """)
    img1 = Image.open('picture/20250711/更新幅.png')
    st.image(img1, caption='FUSION', use_container_width=True)

    st.subheader("実験結果③", divider='rainbow')
    st.markdown("""
    - シグモイドカーネル
      - パラメータチューニング：グリッドサーチ（gamma：[0.01, 0.05, 0.1, 0.2, 0.5],coef0：[-5, -2, 0, 2, 5]）
      - 標準化し内積を計算したデータを見ると、-10〜10に多いことからgamma,coef0を調整した
    """)
    img1 = Image.open('picture/20250711/SIGMOID_FUSION.png')
    img2 = Image.open('picture/20250711/SIGMOID_BSPOP.png')
    img3 = Image.open('picture/20250711/SIGMOID_PAIN.png')
    st.image(img1, caption='FUSION', use_container_width=True)
    st.image(img2, caption="BS-POP", use_container_width=True)
    st.image(img3, caption="PainDITECT", use_container_width=True)

    st.markdown("""
    - 多項式カーネル
      - パラメータチューニング：グリッドサーチ（gamma：[0.01, 0.05, 0.1],coef0：[3, 5, 8],degree：[2, 3]）
      - カーネル値が1〜100に収めたい,内積の値が-10〜10に多いことからgamma,coef0を調整した
      - degreeは、4以上になると急激に値が増加するため、2,3で調整した
    """)
    img1 = Image.open('picture/20250711/POLY_FUSION.png')
    img2 = Image.open('picture/20250711/POLY_BSPOP.png')
    img3 = Image.open('picture/20250711/POLY_PAIN.png')
    st.image(img1, caption='FUSION', use_container_width=True)
    st.image(img2, caption="BS-POP", use_container_width=True)
    st.image(img3, caption="PainDITECT", use_container_width=True)

    st.markdown("""
    - RBFカーネル
    """)
    img1 = Image.open('picture/20250711/RBF_FUSION.png')
    img2 = Image.open('picture/20250711/RBF_BSPOP.png')
    img3 = Image.open('picture/20250711/RBF_PAIN.png')
    st.image(img1, caption='FUSION', use_container_width=True)
    st.image(img2, caption="BS-POP", use_container_width=True)
    st.image(img3, caption="PainDITECT", use_container_width=True)

    st.markdown("""
    - 補足
    """)
    img1 = Image.open('picture/20250711/標準化内積.png')
    st.image(img1, caption='標準化データの内積の分布', use_container_width=True)

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