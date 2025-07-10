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
    - プログラム変更
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
    st.subheader("実験の概要①", divider='rainbow')
    st.markdown("""
        各質問項目に対して適切な重み付けを山登り法で行う  \n
        - 各ステップで特徴量すべてに対して[-ε,0,+ε]の3方向で評価、重みの更新を行う
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
        - 結果の出力：正答率（平均）, 感度 , 特異度, スコアの推移のグラフ
        """)
    with st.container(border=True):
        st.subheader("山登り法", divider='rainbow')
        st.markdown("""
        「今の解よりも良い、今の解に近い解を新しい解にする」ことを繰り返して良い解を求める方法  \n
        （最も代表的な局所探索法として知られている。）
        """)
        img = Image.open('picture/20250523/山登り法.png')
        st.image(img, caption='https://algo-method.com/descriptions/5HVDQLJjbaMvmBL5', use_container_width=True)

    st.subheader("実験の概要②", divider='rainbow')

with st.container(border=True):
    st.subheader("実験結果①", divider='rainbow')
    st.markdown("""
    - シグモイドカーネル
    """)
    img1 = Image.open('picture/20250613/スクリーンショット 2025-06-13 7.56.53.png')
    img2 = Image.open('picture/20250613/0.1重み初期1.png')
    img3 = Image.open('picture/20250613/1初期1_100.png')
    st.image(img1, caption='FUSION', use_container_width=True)
    st.image(img2, caption="BS-POP", use_container_width=True)
    st.image(img3, caption="PainDITECT", use_container_width=True)

    st.subheader("実験結果②", divider='rainbow')
    st.markdown("""
    - シグモイドカーネル
    """)
    img1 = Image.open('picture/20250613/スクリーンショット 2025-06-13 7.56.53.png')
    img2 = Image.open('picture/20250613/0.1重み初期1.png')
    img3 = Image.open('picture/20250613/1初期1_100.png')
    st.image(img1, caption='FUSION', use_container_width=True)
    st.image(img2, caption="BS-POP", use_container_width=True)
    st.image(img3, caption="PainDITECT", use_container_width=True)
    
    st.subheader("実験結果③", divider='rainbow')
    st.markdown("""
    - シグモイドカーネル
    """)
    img1 = Image.open('picture/20250613/スクリーンショット 2025-06-13 7.56.53.png')
    img2 = Image.open('picture/20250613/0.1重み初期1.png')
    img3 = Image.open('picture/20250613/1初期1_100.png')
    st.image(img1, caption='FUSION', use_container_width=True)
    st.image(img2, caption="BS-POP", use_container_width=True)
    st.image(img3, caption="PainDITECT", use_container_width=True)

    st.markdown("""
    - 多項式カーネル
    """)
    img1 = Image.open('picture/20250613/スクリーンショット 2025-06-13 7.56.53.png')
    img2 = Image.open('picture/20250613/0.1重み初期1.png')
    img3 = Image.open('picture/20250613/1初期1_100.png')
    st.image(img1, caption='FUSION', use_container_width=True)
    st.image(img2, caption="BS-POP", use_container_width=True)
    st.image(img3, caption="PainDITECT", use_container_width=True)

    st.markdown("""
    - RBFカーネル
    """)
    img1 = Image.open('picture/20250613/スクリーンショット 2025-06-13 7.56.53.png')
    img2 = Image.open('picture/20250613/0.1重み初期1.png')
    img3 = Image.open('picture/20250613/1初期1_100.png')
    st.image(img1, caption='FUSION', use_container_width=True)
    st.image(img2, caption="BS-POP", use_container_width=True)
    st.image(img3, caption="PainDITECT", use_container_width=True)

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