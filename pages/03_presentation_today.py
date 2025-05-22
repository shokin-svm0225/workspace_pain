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
    st.write("- 前回の内容")
    st.write("- 実験の概要")
    st.write("- 実験結果")
    st.write("- 考察")
    st.write("- 今後の予定の確認")
    st.write("- アドバイス")

with st.container(border=True):
    st.subheader("今までの内容")
    st.write("- 欠損値の補完")
    st.write("--- 欠損値削除 , 中央値補完 , 平均値補完 , k-NN法補完")
    st.write("- 重み付け")
    st.write("--- 判定に影響が多い・少ない質問項目に対して、主観で重み付けを行う（1.5倍,0.5倍）")
    st.write("--- 相関係数を調べ、相関のある質問項目に対して重み付けを行う（1.5倍,0.5倍）")
    st.write("- データの可視化")
    st.write("--- 質問項目ごとのデータの散らばり・外れ値がないかを確認する")
    st.write("- streamlit上で実験をできるようにした")

with st.container(border=True):
    st.subheader("実験の概要", divider='rainbow')
    st.markdown("""
        主観でつけた重み付けで精度を出し、そこから精度の高い方向へ重みを変えていくことを繰り返し、良い重みを求める。  \n
        **準備**
        - 質問表：BS-POP
        - 欠損値の補完：欠損値削除・中央値補完・平均値補完・k-NN法補完
        - 重み付け
          - デフォルト：D1,D2,D7,D8,D9,D11,D13,D14,D17：* 1.5、D6,D10,D18：* 0.5、その他：* 1.0
          - D1から山登り法を繰り返し、どちらに方向を変えても精度が下がったら、次のカラムに移動する
        - カーネル：線形カーネル
        - パラメータチューニング(C)：グリッドサーチ
        - 結果の出力：正答率（平均）, 感度 , 特異度
        """)
    with st.container(border=True):
        st.subheader("山登り法", divider='rainbow')
        st.markdown("""
        「今の解よりも良い、今の解に近い解を新しい解にする」ことを繰り返して良い解を求める方法  \n
        （最も代表的な局所探索法として知られている。）
        """)

with st.container(border=True):
    st.header("実験結果")
    st.write("- 実験1（重み付け：全特徴量 * 1.0）")
    img = Image.open('picture/実験1.png')
    st.image(img, caption='実験1', use_container_width=True)
    st.write("- 実験2（重み付け：D1,D2,D7,D8,D9,D11,D13,D14,D17：* 1.5、D6,D10,D18：* 0.5、その他：* 1.0）")
    img = Image.open('picture/実験2.png')
    st.image(img, caption='実験2', use_container_width=True)
    st.write("- 結果を可視化してみる")
    img = Image.open('picture/質問表ごとの正答率.png')
    st.image(img, caption='質問表ごとの正答率', use_container_width=True)
    img = Image.open('picture/欠損値補完ごとの正答率.png')
    st.image(img, caption='欠損値補完ごとの正答率', use_container_width=True)
    img = Image.open('picture/標準化の有無ごとの正答率.png')
    st.image(img, caption='標準化の有無ごとの正答率', use_container_width=True)

with st.container(border=True):
    st.header("考察")
    st.write("- BS-POPの精度が低いため、重点を置いてデータの加工を行う必要がある（神経障害性疼痛と不明の診断が悪い）")
    st.write("- 不明の精度が毎回低く出ている、データが少ないから？（データオグメンテーションはあり？）")
    st.write("- 主観での重み付けではあまり効果がなかったため、より深く考える必要がある")
    st.write("- 欠損値補完では、どの手法を取っても同じような結果になったが、今後重み付けや特徴量の加工によって変化する可能性あり")
    st.write("- パラメータチューニングを行うと、ほとんど0.01〜0.1の間に収まっているため、0.01〜0.1の間でより細かく区切ってチューニングする必要がある")

with st.container(border=True):
    st.header("今後の予定の確認")
    st.write("- 重み付け")
    st.write("--- データの散らばり具合を可視化")
    st.write("--- 診断に影響がある特徴量を見つける")
    st.write("--- 特徴量の選択：ランダムフォレスト")
    st.write("--- 何かアイデアがあればお願いします")
    st.write("- パラメータの候補の範囲をより細かくしてチューニングする")
    st.write("- 他のカーネルで実験")

