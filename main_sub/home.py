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
    st.title('ホーム')
    st.header("研究内容")
    st.write("AIモデルの一つであるSVMを用い、整形外科での診断補助システム開発を行う")
    st.header("背景")
    st.write("整形外科の患者が訴える典型的症状に、疼痛（慢性痛の総称）がある。人口の20％以上が何らかの疼痛を感じているとの報告もある程、普遍的な症状の一つ。  \n医師による疼痛原因の診断法は複数開発されている一方、手法により精度の差があることが知られている。")
    st.header("目的")
    st.write("人工知能による疼痛診断の自動化を目指す。  \n医学的診断を人工知能で行う際の制約として、「判断根拠を説明できなければいけない」、また「各医療機関が持つデータ量は限定的」というものがある。  \nこれらを解決するため、「少ないデータでも高い精度を達成しやすい」、そして「判断理由の解釈も比較的容易」という特徴を持つ\nSVMを用いることで、高精度かつ説明可能なAIによる診断補助システム構築を目指す。")
    st.header("質問項目の説明")
    # タブの作成
    tab1, tab2, tab3 = st.tabs(["PainDETECT", "BS-POP", "特徴寮拡大"])
    # 各タブに内容を追加
    with tab1:
        # CSVファイルのパスを指定
        csv_file_path_5 = '/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/painditect_質問.csv'  # ファイルパスを指定
        df_paindetect = pd.read_csv(csv_file_path_5)
        # データフレームを表示
        st.dataframe(df_paindetect)

    with tab2:
        # CSVファイルのパスを指定
        csv_file_path_6 = '/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/bspop_質問.csv'  # ファイルパスを指定
        df_bspop = pd.read_csv(csv_file_path_6)
        # データフレームを表示
        st.dataframe(df_bspop)
        # CSVファイルのパスを指定
        csv_file_path_7 = '/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/bspop_痛みの経過.csv'  # ファイルパスを指定
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

    st.header("参考サイト")
    st.write("- [Streamlit_documentation](https://docs.streamlit.io/): streamlitのドキュメント参考")

    st.header("今後の予定")
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

    st.header("実験ログ")
    st.write("- 20241101 : streamlit上で実験ができるようにしました")
    st.write("- 20241102 : 今まで利用したCSVファイルのカラムを統一にしました")
    st.write("- 20241106 : 各質問項目における相関係数の出力をしました")
    st.write("- 20241107 : streamlit上で相関係数の出力・評価の確認を可能にしました")