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
    st.write("- データの可視化")
    st.write("--- 疼痛の種類ごとの各質問項目について(点数の合計人数)図で可視化しました")
    st.write("- データの標準化")
    st.write("- カーネルでの実験")
    st.write("- パラメータチューニング")
    st.write("- 今後の予定の確認")
    st.write("- アドバイス")

    st.header("データの可視化")
    st.write("- 目的変数(痛みの種類)の分布")
    st.write("-- 侵害T = 1 , 神経T = 2 , 不明T = 3")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/typeofpain.png')
    st.image(img, use_container_width=True)
    st.write("分布が大きく偏っていて、不均衡データの可能性がある")
    st.write("機械学習を扱う際の評価指標の選び方に注意する必要がる")

    st.write("- 侵害受容性疼痛と診断されたデータの各質問項目における点数ごとの合計人数の分布")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/侵害受容性疼痛_可視化.png')
    st.image(img, use_container_width=True)
    st.write("P8,P12は0.0に近い点数に合計人数が集まっている傾向があり、診断に影響してそう → 重み付けを高く")
    st.write("D8は他の疼痛と比べて点数の合計人数が均衡している → 重み付けを低く")

    st.write("- 神経障害性疼痛と診断されたデータの各質問項目における点数ごとの合計人数の分布")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/神経障害性疼痛_可視化.png')
    st.image(img, use_container_width=True)
    st.write("P8,P12は他の疼痛と比べて点数の合計人数が均衡している → 重み付けを低く")
    st.write("D8は1.0の点数の合計人数が他の点数より圧倒的に高く、診断に影響してそう → 重み付けを高く")

    st.write("- 不明と診断されたデータの各質問項目における点数ごとの合計人数の分布")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/不明_可視化.png')
    st.image(img, use_container_width=True)
    st.write("P4は1.0の点数が圧倒的に高く、診断に影響してそう → 重み付けを高く")
    st.write("P8,P12は0.0に近い点数に合計人数が集まっている傾向があり、診断に影響してそう → 重み付けを高く")
    st.write("D10は他の疼痛と比べて1.0,2.0の点数が高く、診断に影響してそう → 重み付けを高く")
    
    st.write("- 人数が均衡している質問項目に関しては、重み付けを低くし、偏りのある質問項目は高くするのは違うか？")

    st.header("データの標準化")
    st.write("- 質問項目によって点数の範囲が異なるため、各質問項目の点数を標準化する")
    st.write("- sklearnにあるStandardScalerというライブラリを用いてデータの標準化をする")
    st.write("- 欠損値を平均値で補完したデータを標準化")
    st.write("- 学習データ:70%, テストデータ:30%")
    st.write("- 標準化なし、標準化ありでそれぞれ10回実験し、その平均を精度とする")
    st.write("-- 実験結果")
    st.write("-  標準化なし:約57.97%")
    st.write("-  標準化あり:約45.06%")
    st.write("- 標準化したのに精度が下がったのは何故か？")

    st.header("ハイパーパラメータ(C)のチューニング")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/12:13/1213_svm_parameter.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, use_container_width=True)
    st.write("--- svm.setKernel(cv2.ml.SVM_LINEAR)：カーネル関数の種類を表す")
    st.write("--- svm.setGamma(1):LINEAR以外のカーネルの場合用いるパラメータ")
    st.write("--- svm.setC(1)：どれだけ誤分類を許容するかについてのパラメータ")
    st.write("--- svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1500, 1.0e-06))：終了条件")
    st.write("- グリッドサーチ")
    st.write("--- 指定された範囲の中で、すべての組み合わせを総当たりで探索して最適なパラメータを設定する手法")
    body_4 = """
        # パラメータの候補を設定
        gamma_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]  # 0.1から100までの範囲、ステップ幅1
        C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

        best_score = 0
        best_params = None

        # グリッドサーチを実行(複数のリストの直積)
        for gamma, C in itertools.product(gamma_values, C_values):
            svm = cv2.ml.SVM_create()
            svm.setType(cv2.ml.SVM_C_SVC)
            svm.setKernel(cv2.ml.SVM_LINEAR)
            svm.setGamma(gamma)
            svm.setC(C)
            svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1500, 1.0e-06))

            # トレーニング
            svm.train(datas, cv2.ml.ROW_SAMPLE, labels)

            # テストデータで評価
            _, predicted = svm.predict(test_datas)
            score = np.sum(test_labels == predicted.flatten()) / len(test_labels)

            # ベストスコアの更新
            if score > best_score:
                best_score = score
                best_params = {"gamma": gamma, "C": C}

            st.write(f"Gamma: {gamma}, C: {C}, Score: {score:.4f}")

            # モデルを保存
            svm.save(SAVE_TRAINED_DATA_PATH)

        st.write("最適なパラメータ:", best_params)
        st.write("最高スコア:", best_score)
        """
    st.code(body_4, language="python")

    st.header("カーネルでの実験")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/12:13/1213_svm_parameter.png')
    st.image(img, use_container_width=True)
    st.write("--- svm.setKernel(cv2.ml.SVM_LINEAR)：カーネル関数の種類を表す")
    st.write("- svm.setKernel(cv2.ml.SVM_LINEAR)：線形カーネル")
    st.write("- svm.setKernel(cv2.ml.SVM_POLY)：多項式カーネル")
    st.write("- svm.setKernel(cv2.ml.SVM_RBF)：ガウス(RBF)カーネル")
    st.write("- svm.setKernel(cv2.ml.SVM_SIGMOID)：シグモイドカーネル")
    st.write("--- svm.setGamma(1):LINEAR以外のカーネルの場合用いるパラメータ")
    st.write("- svm.setDegree(),svm.setCoef0()などあるが、何を示しているのかわかりません")
    st.write("--- svm.setC(1)：どれだけ誤分類を許容するかについてのパラメータ")
    st.write("--- svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1500, 1.0e-06))：終了条件")
    st.write("-- 実験結果")
    st.write("- 標準化なし、標準化ありでそれぞれ10回実験し、その平均を精度とする")
    body_45= """
        # パラメータの候補を設定
        gamma_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] 
        C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        D_values = [1, 2, 3, 4, 5, 6, 8, 10]
        r_values = [1, 3, 5, 7, 10, 15, 20]
        """
    st.code(body_45, language="python")
    st.write("- 線形カーネル")
    st.write("--- 標準化なし:約53.02%, 標準化あり:約55.96%")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/スクリーンショット 2025-01-24 7.19.12.png')
    st.image(img, caption='線形カーネル', use_container_width=True)
    st.write("- シグモイドカーネル")
    st.write("--- 標準化なし:約60.99%, 標準化あり:約53.76%")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/スクリーンショット 2025-01-24 7.45.28.png')
    st.image(img, caption='シグモイドカーネル', use_container_width=True)
    st.write("- 多項式カーネル")
    st.write("--- 標準化なし:約6.64%, 標準化あり:約51.62%")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/スクリーンショット 2025-01-24 7.41.16.png')
    st.image(img, caption='多項式カーネル', use_container_width=True)
    st.write("- RBFカーネル")
    st.write("--- 標準化なし:約6.64%, 標準化あり:61%")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/スクリーンショット 2025-01-24 7.37.53.png')
    st.image(img, caption='RBFカーネル', use_container_width=True)
    st.write("- 様々なカーネルの中でもシグモイドカーネルが標準化なしのデータで一番良い結果が出た")
    st.write("- しかし、でれも精度は高くなく、パラメータチューニングやデータエンジニアリングでの工夫が必要である")
    st.write("- 多項式カーネル、RBFカーネルではコードのミスの可能性あり（再度確認）")

    st.header("今後の予定の確認")
    st.write("- 特徴量エンジニアリング")
    st.write("--- 最初からデータセットを細部まで確認して特性を知る")
    st.write("- 特徴量の選択・重要度の評価/")
    st.write("--- ランダムフォレスト・特徴選択:RFE（再帰的特徴消去）")
    st.write("- ハイパーパラメータ(C)のチューニング")
    st.write("- 他の機械学習モデルでの実験")
    st.write("- streamlitのUIデザインの変更")
