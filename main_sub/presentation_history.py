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
    st.title('これまでの発表内容')
    
    #データの加工方法の指定
    options = ['11月8日', '11月22日', '12月13日', '1月24日', '4月25日']

    # セレクトボックスを作成し、ユーザーの選択を取得
    data_processing = st.selectbox('ご覧になる発表内容の日程を選択してください', options, index = None, placeholder="選択してください")
    
    # 選択されたオプションに応じた処理
    if data_processing == "11月8日":
        st.header("11月8日")
        st.write("データから偏相関係数を求め、重み付けを行いました")
        st.write("- 訓練データ：70%,テストデータ：30%")
        st.write("- FUSIONのデータセットで実行")
        st.write("- エクセル上でカラースケールを用いて可視化")
        st.write("- 重み付け：偏相関係数の高いカラムに1.5倍、低いカラムに0.5倍で実行")
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/pain_exper.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img, caption='PainDETECT', use_container_width=True)

        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/bspop_exper.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img, caption='BS-POP', use_container_width=True)

        st.header("実験前")
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/実験前_omomi.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img, use_container_width=True)

        st.header("実験後")
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/実験後_omomi.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img, use_container_width=True)
        st.write("- 何度か実験を行ったが、実験前と変化は見られなかった")
        st.write("- 極端に大きく、小さくしたが、変化は見られなかった（若干下がったかも）")

    elif data_processing == "11月22日":
        st.header("11月22日")
        st.write("- UIデザインの変更")
        st.write("--- 実験の重み付けの指定の部分を、縦ではなく横に出力するように変更")
        st.write("--- サイドバーをプルダウン式ではなく、一覧を表示するようにした")
        st.write("--- 実験の使用するカラムの指定を全選択するようにした")
        st.write("- 目的変数(痛みの種類)を含めて相関係数を求めて重み付けを行った")

        st.header("目的変数(痛みの種類)を含めて相関係数を求めて重み付け")
        st.write("目的変数(痛みの種類)と説明変数(各質問項目)との相関係数を求め、関係性の強さを示し、結果を元に重み付けと特徴量選択をして実験を行った")
        st.write("- 相関係数の計算に用いたデータ：２つの質問表を組み合わせたもの")
        st.write("- 痛みの種類：侵害T = 0 , 神経T = 1 , 不明T = 2")
        st.markdown('#### 相関係数')
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/coeff_pain.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img, use_container_width=True)
        st.write("- 比較的に高い相関が見られなかった")
        st.write("- 相対的に見て負の相関がない、重み付けの判定がしづらい")

        st.markdown('#### 重み付け')
        st.write("- 相関係数の高いカラムに1.5倍、低いカラムに0.5倍で実行")
        st.write("- 1.5倍：P8,P12")
        st.write("- 0.5倍：P9,D10,D14,D15,D17")
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_weight_null.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="欠損値削除" , use_container_width=True)
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_weight_median.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="中央値補完" , use_container_width=True)
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_weight_mean.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="平均値補完" , use_container_width=True)
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_weight_knn.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="k-NN法補完" , use_container_width=True)
        st.write("結果・考察")
        st.write("- 様々な欠損値の補完データで試したが、どれも結果にばらつきがり、かつ精度が上がらなかった")

        st.markdown('#### 特徴量選択')
        st.write("- 相関係数の高いカラム、低いカラムを選択し、使用する特徴量を圧縮して実験をする")
        st.write("- 使用するカラム：P8,P9,P12,D10,D14,D15,D17")
        st.write("- 重み付けは、上記と同様に行う")
        st.write("- 0.5倍：P9,D10,D14,D15,D17")
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_select_null.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="欠損値削除" , use_container_width=True)
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_select_median.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="中央値補完" , use_container_width=True)
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_select_mean.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="平均値補完" , use_container_width=True)
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_select_knn.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="k-NN法補完" , use_container_width=True)
        st.write("結果・考察")
        st.write("- どの欠損値補完データにおいても約70%ほどに収束した")
        st.write("- ほんとに適切に特徴量選択できたのか他の選択で試す必要がある")

        st.header("今後の予定")
        st.write("- 使用するカラムの指定から重み付けを行えるスライドバーの設定")
        st.write("- streamlitのプログラムのソフトコーディング化")
        st.write("- スケーリング（データの正規化・標準化）")
        st.write("- 特徴量重要度の評価")
        st.write("--- 特徴選択：RFE（再帰的特徴消去）")
        st.write("--- RFE：最も特徴量重要度が低いものを削除することを繰り返し、指定した数まで消去をする手法")

    elif data_processing == "12月13日":
        st.title('発表内容')
        st.header("概要")
        st.write("- UIデザインの変更")
        st.write("--- 今までの発表内容(stramlitで作成したもの)を選択で見れるようにした")
        st.write("--- 実験の使用するカラムの重み付けをスライダーで設定できるようにした")
        st.write("- 今後の予定の確認")
        st.write("- アドバイス")

        st.header("UIデザインの変更")
        st.write("実際に動かして確認")
        st.write("- 今までの発表内容(stramlitで作成したもの)を選択で見れるようにした")
        st.write("- 実験の使用するカラムの重み付けをスライダーで設定できるようにした")

        st.header("今後の予定の確認")
        st.write("- 特徴量エンジニアリング")
        st.write("- 特徴量重要度の評価")
        st.write("- ハイパーパラメータ(C)のチューニング")
        st.write("- その他")


        st.header("特徴量エンジニアリング")
        st.write("- 目的変数(痛みの種類)の数値化の見直し")
        st.write("--- 前回、侵害T = 0 , 神経T = 1 , 不明T = 2とおいたが、上手く分類できていないのではないか")
        st.write("--- 自然においてあげるなら、侵害T = -1 , 神経T = 1 , 不明T = 0 でおく")
        st.write("--- Word2vecで数値化する")
        st.write("- 重み付けを行うカラムと値の見直し")
        st.write("--- 前回、相対的に負の相関にあるカラムを0.5倍にしていた")
        st.write("- スケーリング（データの正規化・標準化）")

        st.header("Word2vecによる数値化")
        st.write("- 文章に含まれる単語を「数値ベクトル」に変換し、その意味を把握していくという自然言語処理の手法")
        st.write("- 語句のデータを学習させ、その中から3つの痛みの種類の数値化を行うという認識で合っているか？")
        st.write("- 数値化すると多次元のベクトルに変換されると予想されるが、一次元でなくでも大丈夫か？")
        st.write("- [Word2vecの説明](https://aismiley.co.jp/ai_news/word2vec/): 参考文献")
        st.write("- [北村さんから共有していただいた動画](https://youtu.be/sK3HqLwag_w?si=VlkOHj8PZeTzUJEM): 参考文献")



        st.header("特徴量重要度の評価")
        st.write("- 前回、相関係数による特徴量の選択と重み付けの設定を行った → 強い相関が見られず、正確な選択が難しい")
        st.write("- 特徴選択：RFE（再帰的特徴消去）Recursive Feature Elimination の略")
        st.write("--- RFE：最も特徴量重要度が低いものを削除することを繰り返し、指定した特徴量の数まで消去をする手法")

        st.header("ハイパーパラメータ(C)のチューニング")
        st.write("- クロスバリデーション（交差検証）")
        st.write("--- モデル性能の評価は行ったが、ハイパーパラメータのチューニングはできるのか？")
        st.write("- グリッドサーチ")
        st.write("--- 指定された範囲の中で、すべての組み合わせを総当たりで探索して最適なパラメータを設定する手法")
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/12:13/1213_svm_parameter.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img, use_container_width=True)
        st.write("--- svm.setKernel(cv2.ml.SVM_LINEAR)：カーネル関数の種類を表す")
        st.write("--- svm.setGamma(1)：LINEAR以外のカーネルの場合用いるパラメータ")
        st.write("--- svm.setC(1)：どれだけ誤分類を許容するかについてのパラメータ")
        st.write("--- svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1500, 1.0e-06))：終了条件")
        st.write("--- メリット：重複と漏れがない")
        st.write("--- デメリット：広い範囲の探索は厳しい")
        st.write("- ランダムサーチ")
        st.write("--- 指定された回数の中で無作為に探索して最適なパラメータを設定する手法")
        st.write("--- メリット：探索の順番がランダムであるため、広い範囲を満遍なく探索できる")
        st.write("--- デメリット：重複や最適なパラメータを見逃す場合がある")
        st.write("- ベイズ最適化")
        st.write("--- ベイズ最適化を使ったアルゴリズムによる自動探索")
        st.write("--- ガウス過程といった確率モデルを用いて過去の探索履歴を考慮して、次に探索すべきハイパラを合理的に選択する")
        st.write("- OpenCVのSVMモデルが記載されているサイトあれば教えてほしい")
        st.header("その他")
        st.write("- データ分析")
        st.write("--- 最初からデータセットを細部まで確認して特性を知る")
        st.write("- 他の機械学習モデルでの実験")
        st.write("- streamlitのUIデザインの変更")
    
    elif data_processing == "1月24日":
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

    elif data_processing == "4月25日":
        st.header("何やってるんですか、勉強してください")
        gif_path = "/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/kounogento.gif"
        st.image(gif_path,caption="河野玄斗")