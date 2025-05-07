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

def show():
    st.title('発表内容')
    st.header("概要")
    st.write("- 今までの内容")
    st.write("- 実験の概要")
    st.write("- 実験結果")
    st.write("- 考察")
    st.write("- 今後の予定の確認")
    st.write("- アドバイス")

    st.header("今までの内容")
    st.write("- 欠損値の補完")
    st.write("--- 欠損値削除 , 中央値補完 , 平均値補完 , k-NN法補完")
    st.write("- 重み付け")
    st.write("--- 判定に影響が多い・少ない質問項目に対して、主観で重み付けを行う（1.5倍,0.5倍）")
    st.write("--- 相関係数を調べ、相関のある質問項目に対して重み付けを行う（1.5倍,0.5倍）")
    st.write("- データの可視化")
    st.write("--- 質問項目ごとのデータの散らばり・外れ値がないかを確認する")
    st.write("- streamlit上で実験をできるようにした")

    st.header("実験の概要")
    st.write("準備")
    st.write("- 質問表：PainDITECT・BS-POP・FUSION")
    st.write("- 欠損値の補完：欠損値削除・中央値補完・平均値補完・k-NN法補完")
    st.write("- 標準化の有無")
    st.write("- 重み付け")
    st.write("--- 全特徴量 * 1.0")
    st.write("--- D1,D2,D7,D8,D9,D11,D13,D14,D17：* 1.5、D6,D10,D18：* 0.5、その他：* 1.0")
    st.write("--- PainDITECTは、侵害受容性疼痛の診断に優れており、それに関する質問項目が多く、主観で重み付けがしづらかった。(11/8)")
    st.write("- カーネル：線形カーネル")
    st.write("- パラメータチューニング(C)：グリッドサーチ")
    st.write("--- パラメータの候補範囲：0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000")
    st.write("- 結果の出力：正答率（平均）, 感度 , 特異度")
    st.write("--- 感度：疾患のある人を正しく陽性と判定できるか（真陽性）")
    st.write("--- 特異度：疾患のない人を正しく陰性と判定できるか（真陰性）")
    st.write("内容")
    st.write("- 5-分割交差検証を行い、候補のパラメータごとの平均スコア（正答率）をそれぞれ計算し、最も良かったパラメータとそのスコア（正答率）を出力する")
    st.write("- 交差検証：汎化性能を評価する統計的な手法")
    st.write("- k-交差検証：データをK個に分割してそのうち1つをテストデータに残りのK-1個を学習データとして正解率の評価を行う。これをK個のデータすべてが1回ずつテストデータになるようにK回学習を行なって精度の平均をとる手法である。")

    st.header("実験結果")
    st.write("- 実験1（重み付け：全特徴量 * 1.0）")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/実験1.png')
    st.image(img, caption='実験1', use_container_width=True)
    st.write("- 実験2（重み付け：D1,D2,D7,D8,D9,D11,D13,D14,D17：* 1.5、D6,D10,D18：* 0.5、その他：* 1.0）")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/実験2.png')
    st.image(img, caption='実験2', use_container_width=True)
    st.write("- 結果を可視化してみる")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/質問表ごとの正答率.png')
    st.image(img, caption='質問表ごとの正答率', use_container_width=True)
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/欠損値補完ごとの正答率.png')
    st.image(img, caption='欠損値補完ごとの正答率', use_container_width=True)
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/標準化の有無ごとの正答率.png')
    st.image(img, caption='標準化の有無ごとの正答率', use_container_width=True)

    st.header("考察")
    st.write("- BS-POPの精度が低いため、重点を置いてデータの加工を行う必要がある（神経障害性疼痛と不明の診断が悪い）")
    st.write("- 不明の精度が毎回低く出ている、データが少ないから？（データオグメンテーションはあり？）")
    st.write("- 主観での重み付けではあまり効果がなかったため、より深く考える必要がある")
    st.write("- 欠損値補完では、どの手法を取っても同じような結果になったが、今後重み付けや特徴量の加工によって変化する可能性あり")
    st.write("- パラメータチューニングを行うと、ほとんど0.01〜0.1の間に収まっているため、0.01〜0.1の間でより細かく区切ってチューニングする必要がある")

    st.header("今後の予定の確認")
    st.write("- 重み付け")
    st.write("--- データの散らばり具合を可視化")
    st.write("--- 診断に影響がある特徴量を見つける")
    st.write("--- 特徴量の選択：ランダムフォレスト")
    st.write("--- 何かアイデアがあればお願いします")
    st.write("- パラメータの候補の範囲をより細かくしてチューニングする")
    st.write("- 他のカーネルで実験")
