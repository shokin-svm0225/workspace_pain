import streamlit as st
import itertools
import plotly.express as px
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
from streamlit_option_menu import option_menu

st.title('発表履歴')

#データの加工方法の指定
options = ['2024年11月8日', '2024年11月22日', '2024年12月13日', '2025年1月24日', '2025年4月25日', '2025年5月23日', '2025年6月13日', '2025年7月11日', '2025年10月17日']

# セレクトボックスを作成し、ユーザーの選択を取得
data_processing = st.sidebar.selectbox('日程を選択してください', options, index = None, placeholder="選択してください")

# 選択されたオプションに応じた処理
if data_processing == "2024年11月8日":
    st.subheader('2024年11月8日', divider='rainbow')
    st.write("データから偏相関係数を求め、重み付けを行いました")
    st.write("- 訓練データ：70%,テストデータ：30%")
    st.write("- FUSIONのデータセットで実行")
    st.write("- エクセル上でカラースケールを用いて可視化")
    st.write("- 重み付け：偏相関係数の高いカラムに1.5倍、低いカラムに0.5倍で実行")
    img = Image.open('picture/pain_exper.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, caption='PainDETECT', use_container_width=True)

    img = Image.open('picture/bspop_exper.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, caption='BS-POP', use_container_width=True)

    st.header("実験前")
    img = Image.open('picture/実験前_omomi.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, use_container_width=True)

    st.header("実験後")
    img = Image.open('picture/実験後_omomi.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, use_container_width=True)
    st.write("- 何度か実験を行ったが、実験前と変化は見られなかった")
    st.write("- 極端に大きく、小さくしたが、変化は見られなかった（若干下がったかも）")

elif data_processing == "2024年11月22日":
    st.subheader('2024年11月22日', divider='rainbow')
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
    img = Image.open('picture/11:22/coeff_pain.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, use_container_width=True)
    st.write("- 比較的に高い相関が見られなかった")
    st.write("- 相対的に見て負の相関がない、重み付けの判定がしづらい")

    st.markdown('#### 重み付け')
    st.write("- 相関係数の高いカラムに1.5倍、低いカラムに0.5倍で実行")
    st.write("- 1.5倍：P8,P12")
    st.write("- 0.5倍：P9,D10,D14,D15,D17")
    img = Image.open('picture/11:22/11_22_weight_null.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img,caption="欠損値削除" , use_container_width=True)
    img = Image.open('picture/11:22/11_22_weight_median.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img,caption="中央値補完" , use_container_width=True)
    img = Image.open('picture/11:22/11_22_weight_mean.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img,caption="平均値補完" , use_container_width=True)
    img = Image.open('picture/11:22/11_22_weight_knn.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img,caption="k-NN法補完" , use_container_width=True)
    st.write("結果・考察")
    st.write("- 様々な欠損値の補完データで試したが、どれも結果にばらつきがり、かつ精度が上がらなかった")

    st.markdown('#### 特徴量選択')
    st.write("- 相関係数の高いカラム、低いカラムを選択し、使用する特徴量を圧縮して実験をする")
    st.write("- 使用するカラム：P8,P9,P12,D10,D14,D15,D17")
    st.write("- 重み付けは、上記と同様に行う")
    st.write("- 0.5倍：P9,D10,D14,D15,D17")
    img = Image.open('picture/11:22/11_22_select_null.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img,caption="欠損値削除" , use_container_width=True)
    img = Image.open('picture/11:22/11_22_select_median.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img,caption="中央値補完" , use_container_width=True)
    img = Image.open('picture/11:22/11_22_select_mean.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img,caption="平均値補完" , use_container_width=True)
    img = Image.open('picture/11:22/11_22_select_knn.png')
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

elif data_processing == "2024年12月13日":
    st.subheader('2024年12月13日', divider='rainbow')
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
    img = Image.open('picture/12:13/1213_svm_parameter.png')
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

elif data_processing == "2025年1月24日":
    st.subheader('2025年1月24日', divider='rainbow')
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
    img = Image.open('picture/typeofpain.png')
    st.image(img, use_container_width=True)
    st.write("分布が大きく偏っていて、不均衡データの可能性がある")
    st.write("機械学習を扱う際の評価指標の選び方に注意する必要がる")

    st.write("- 侵害受容性疼痛と診断されたデータの各質問項目における点数ごとの合計人数の分布")
    img = Image.open('picture/侵害受容性疼痛_可視化.png')
    st.image(img, use_container_width=True)
    st.write("P8,P12は0.0に近い点数に合計人数が集まっている傾向があり、診断に影響してそう → 重み付けを高く")
    st.write("D8は他の疼痛と比べて点数の合計人数が均衡している → 重み付けを低く")

    st.write("- 神経障害性疼痛と診断されたデータの各質問項目における点数ごとの合計人数の分布")
    img = Image.open('picture/神経障害性疼痛_可視化.png')
    st.image(img, use_container_width=True)
    st.write("P8,P12は他の疼痛と比べて点数の合計人数が均衡している → 重み付けを低く")
    st.write("D8は1.0の点数の合計人数が他の点数より圧倒的に高く、診断に影響してそう → 重み付けを高く")

    st.write("- 不明と診断されたデータの各質問項目における点数ごとの合計人数の分布")
    img = Image.open('picture/不明_可視化.png')
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
    img = Image.open('picture/12:13/1213_svm_parameter.png')
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
    img = Image.open('picture/12:13/1213_svm_parameter.png')
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
    img = Image.open('picture/スクリーンショット 2025-01-24 7.19.12.png')
    st.image(img, caption='線形カーネル', use_container_width=True)
    st.write("- シグモイドカーネル")
    st.write("--- 標準化なし:約60.99%, 標準化あり:約53.76%")
    img = Image.open('picture/スクリーンショット 2025-01-24 7.45.28.png')
    st.image(img, caption='シグモイドカーネル', use_container_width=True)
    st.write("- 多項式カーネル")
    st.write("--- 標準化なし:約6.64%, 標準化あり:約51.62%")
    img = Image.open('picture/スクリーンショット 2025-01-24 7.41.16.png')
    st.image(img, caption='多項式カーネル', use_container_width=True)
    st.write("- RBFカーネル")
    st.write("--- 標準化なし:約6.64%, 標準化あり:61%")
    img = Image.open('picture/スクリーンショット 2025-01-24 7.37.53.png')
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

elif data_processing == "2025年4月25日":
    st.subheader('2025年4月25日', divider='rainbow')
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

elif data_processing == "2025年5月23日":
    st.subheader('2025年5月23日')
    with st.container(border=True):
        st.subheader("アジェンダ", divider='rainbow')
        st.markdown("""
        - プロスラムの変更
        - 前回の内容
        - 実験の概要
        - 実験結果
        - 考察
        - 今後の予定の確認
        - アドバイス
        """)  

    with st.container(border=True):
        st.subheader("プログラムの変更", divider='rainbow')
        st.markdown("""
            - SVMのライブラリをOpenCVからscikit-learnに変更
            """)  
        with st.container(border=True):
            st.subheader('scikit-learn', divider='rainbow')
            st.markdown("""
            Python で利用できるデータ分析や機械学習のためのライブラリの一つ

            - 機械学習のプロジェクト全体を一つのライブラリで管理することが可能
                - データの前処理、教師あり学習、教師なし学習、モデル選択、評価など
            - 非常に充実したドキュメンテーションがある
                - [Scikit-learn_documentation](https://scikit-learn.org/stable/user_guide.html): scikit-learnのドキュメント参考")
            """)
            st.markdown("**SVM（サポートベクトルマシン）**")
            body_1 = """
            class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', 
                coef0=0.0, shrinking=True, probability=False, tol=0.001, 
                cache_size=200, class_weight=None, verbose=False, max_iter=- 1, 
                decision_function_shape='ovr', break_ties=False, random_state=None)
            """
            st.code(body_1, language="python")
            st.markdown("**特徴量X, クラスyを学習データとして学習する**")
            body_2 = """
            fit(X,y)
            """
            st.code(body_2, language="python")
            st.markdown("**テストデータXに対するクラスの予測結果を出力する**")
            body_3 = """
            predict(X)
            """
            st.code(body_3, language="python")
            st.markdown("**K-分割交差検証**")
            body_4 = """
            class sklearn.model_selection.StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)
            """
            st.code(body_4, language="python")
            st.markdown("**標準化**")
            body_5 = """
            class sklearn.preprocessing.StandardScaler(*, copy=True, with_mean=True, with_std=True)
            """
            st.code(body_5, language="python")
            st.markdown("**データを学習用とテスト用に分割する**")
            body_6 = """
            sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
            """
            st.code(body_6, language="python")
            
    with st.container(border=True):
        st.subheader("実験の概要", divider='rainbow')
        st.markdown("""
            主観でつけた重み付けで精度を出し、そこから神経障害性疼痛に関する感度の高い方向へ重みを変えていくことを繰り返し、良い重みを求める。  \n
            **準備**
            - 質問表：BS-POP
            - 欠損値の補完：欠損値削除・中央値補完・平均値補完・k-NN法補完
            - 重み付け
            - デフォルト：D1,D2,D7,D8,D9,D11,D13,D14,D17：* 1.5、D6,D10,D18：* 0.5、その他：* 1.0
            - 重み付けの変更は、1.0刻みで行うとする
            - D1から山登り法を繰り返し、どちらに方向を変えても神経障害性疼痛に関する感度が下がったら、次のカラムに移動する
            - どちらも上がったら比較して感度が高い重みを解とする
            - カーネル：線形カーネル
            - 評価：5-分割交差検証
            - パラメータチューニング(C)：グリッドサーチ
            - 結果の出力：正答率（平均）, 感度 , 特異度
            """)
        with st.container(border=True):
            st.subheader("山登り法", divider='rainbow')
            st.markdown("""
            「今の解よりも良い、今の解に近い解を新しい解にする」ことを繰り返して良い解を求める方法  \n
            （最も代表的な局所探索法として知られている。）
            """)
            img = Image.open('picture/20250523/山登り法.png')
            st.image(img, caption='https://algo-method.com/descriptions/5HVDQLJjbaMvmBL5', use_container_width=True)

    with st.container(border=True):
        st.subheader("実験結果", divider='rainbow')
        st.markdown("""
        - 欠損値削除
        """)
        img = Image.open('picture/20250523/20250523_欠損値削除.png')
        st.image(img, caption='欠損値削除', use_container_width=True)
        st.markdown("""
        - 中央値補完
        """)
        img = Image.open('picture/20250523/20250523_中央値.png')
        st.image(img, caption='中央値補完', use_container_width=True)
        st.markdown("""
        - 平均値補完
        最初に34.62%と比較的大きい値が出てしまい、その後更新なし
        """)
        img = Image.open('picture/20250523/20250523平均値.png')
        st.image(img, caption='平均値補完', use_container_width=True)
        st.markdown("""
        - k-NN法補完   
        """)
        img = Image.open('picture/20250523/20250523_knn.png')
        st.image(img, caption='k-NN法補完', use_container_width=True)

    with st.container(border=True):
        st.subheader("考察", divider='rainbow')
        st.markdown("""
        - 交差検証で評価をしているものの何回か実行すると離れた結果になる
        - 平均値補完では、最初に34.62%と比較的大きい値が出てしまい、その後更新なし
        - 主観で重み付けを行うのは厳しいかな
        - 制度に差があるのは、データ数が少ないから？
        - 侵害需要性疼痛と診断されたデータと神経障害性疼痛と診断されたデータの回答傾向を可視化し、神経障害性疼痛をうまく識別する特徴量が何かを探して重み付けをするのはどうか
        """)

    with st.container(border=True):
        st.subheader("今後の予定の確認", divider='rainbow')
        st.markdown("""
        - 重み付け
        - 山登り法の実装
        - 侵害需要性疼痛と診断されたデータと神経障害性疼痛と診断されたデータの回答傾向を可視化
        - 他に案があれば教えてください
        - 主成分分析で特徴量削減（次元削減）
        """)

elif data_processing == "2025年6月13日":
    st.subheader('2025年6月13日')
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
            st.subheader('プログラムの概要', divider='rainbow')
            # タブの作成
            tab1, tab2, tab3 = st.tabs(["山登り法([-ε,0,+ε]の3方向)", "山登り法([-ε,+ε]の2方向)", "SVM(交差検証)"])
            # 各タブに内容を追加
            with tab1:
                body_1 = """
                # 初期重みを [-5, 5) の範囲でランダムに設定
                initial_weights = np.random.randint(-5, 5, datas.shape[1])

                # 山登り法（1つのCに対して最適な重みを探索）
                def hill_climbing(datas, labels, C, initial_weights, max_iter_1=100, step_size=1):
                    n_features = datas.shape[1]
                    # weights_change = np.ones(n_features)
                    weights_change = initial_weights.copy()  # 外から渡された固定の初期重み
                    st.write("✅ 初期重み:" + str(weights_change.tolist()))

                    best_score, best_X_val, best_y_val, best_pred = evaluate(weights_change, datas, labels, C, return_best_split=True)
                    best_weights = weights_change.copy()


                    # Streamlitの進捗バーとスコア表示
                    hill_bar = st.progress(0)
                    score_history = [best_score]


                    for i in range(max_iter_1):
                        step_best_score = best_score
                        step_best_weights = weights_change.copy()
                        step_best_X_val, step_best_y_val, step_best_pred = best_X_val, best_y_val, best_pred

                        for idx in range(n_features):
                            for delta in [-step_size, 0, step_size]:
                                trial_weights = weights_change.copy()
                                trial_weights[idx] += delta #idx番目の特徴量だけ delta 分変化させた新しい重みを作成

                                # delta = 0 のときは再評価せず、現在のベストスコアと検証結果をそのまま使用
                                if delta == 0:
                                    score = best_score
                                    X_val_tmp, y_val_tmp, pred_tmp = best_X_val, best_y_val, best_pred
                                else:
                                    score, X_val_tmp, y_val_tmp, pred_tmp = evaluate(
                                        trial_weights, datas, labels, C, return_best_split=True
                                    )

                                # 各ステップの中、もっとも良いスコアが得られた場合は、その情報を更新・記録
                                if score > step_best_score:
                                    step_best_score = score
                                    step_best_weights = trial_weights.copy()
                                    step_best_X_val = X_val_tmp
                                    step_best_y_val = y_val_tmp
                                    step_best_pred = pred_tmp

                        # 一番良かった方向へ重みを更新し、スコアや予測結果を上書き
                        weights_change = step_best_weights
                        best_weights = weights_change.copy()
                        best_score = step_best_score
                        best_X_val, best_y_val, best_pred = step_best_X_val, step_best_y_val, step_best_pred

                        # スコア履歴に今回のベストスコアを追加
                        score_history.append(best_score)
                        percent = int((i + 1) / max_iter_1 * 100)
                        hill_bar.progress(percent, text=f"進捗状況{percent}%")

                    return best_weights, best_score, best_X_val, best_y_val, best_pred, score_history
                """
                st.code(body_1, language="python")

            with tab2:
                body_2 = """
                # 初期重みを [-5, 5) の範囲でランダムに設定
                initial_weights = np.random.randint(-5, 5, datas.shape[1])

                # 山登り法で、1つのCに対して最適な特徴量の重みベクトルを探索
                def hill_climbing(datas, labels, C, initial_weights, max_iter_1=100, step_size=0.1):
                    n_features = datas.shape[1]

                    # 初期重みをコピー（元の初期重みは他のC値でも使えるように保持）
                    weights_change = initial_weights.copy()
                    st.write("✅ 初期重み:" + str(weights_change.tolist()))

                    # 初期重みに対するスコアと検証データの情報を取得
                    best_score, best_X_val, best_y_val, best_pred = evaluate(
                        weights_change, datas, labels, C, return_best_split=True
                    )
                    best_weights = weights_change.copy()

                    # Streamlit の進捗バーを初期化
                    hill_bar = st.progress(0)
                    score_history = [best_score]  # スコアの履歴を保存

                    # hill climbing を max_iter_1 回繰り返す
                    for i in range(max_iter_1):
                        # 各ステップで最良のスコアを探す
                        step_best_score = best_score
                        step_best_weights = weights_change.copy()
                        step_best_X_val, step_best_y_val, step_best_pred = best_X_val, best_y_val, best_pred

                        # 各特徴量に対して ±step_size 変更を試す
                        for idx in range(n_features):
                            for delta in [-step_size, step_size]:
                                trial_weights = weights_change.copy()  # 現在の重みをコピー
                                trial_weights[idx] += delta            # 一つの特徴量だけを変更

                                # 新しい重みでモデルを評価
                                score, X_val_tmp, y_val_tmp, pred_tmp = evaluate(
                                    trial_weights, datas, labels, C, return_best_split=True
                                )

                                # スコアが改善されたら、その重みを保存
                                if score > step_best_score:
                                    step_best_score = score
                                    step_best_weights = trial_weights.copy()
                                    step_best_X_val = X_val_tmp
                                    step_best_y_val = y_val_tmp
                                    step_best_pred = pred_tmp

                        # ステップ内で最良だった重みを採用（更新）
                        weights_change = step_best_weights
                        best_weights = weights_change.copy()
                        best_score = step_best_score
                        best_X_val, best_y_val, best_pred = step_best_X_val, step_best_y_val, step_best_pred

                        # スコア履歴の更新と進捗表示
                        score_history.append(best_score)
                        percent = int((i + 1) / max_iter_1 * 100)
                        hill_bar.progress(percent, text=f"進捗状況{percent}%")

                    # 最終的に見つかった最良の重みとスコア、検証結果、スコアの推移を返す
                    return best_weights, best_score, best_X_val, best_y_val, best_pred, score_history
                """
                st.code(body_2, language="python")

            with tab3:
                body_3 = """
                # 重みをかける関数
                def apply_weights(datas, weights_change):
                    return datas * weights_change

                # 指定された重みで交差検証精度を返す関数
                def evaluate(weights_change, datas, labels, C, k=5, return_best_split=False):
                    X_weighted = apply_weights(datas, weights_change)
                    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                    scores = []

                    best_fold_score = 0
                    best_X_val, best_y_val, best_pred = None, None, None

                    for train_index, val_index in skf.split(X_weighted, labels):
                        X_train, X_val = X_weighted[train_index], X_weighted[val_index]
                        y_train, y_val = labels[train_index], labels[val_index]

                        model = SVC(C=C, kernel='linear', max_iter=1500)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        acc = np.mean(y_pred == y_val)
                        scores.append(acc)

                        # 評価指標が最高のfoldを保存
                        if return_best_split and acc > best_fold_score:
                            best_fold_score = acc
                            best_X_val = X_val
                            best_y_val = y_val
                            best_pred = y_pred

                    if return_best_split:
                            return np.mean(scores), best_X_val, best_y_val, best_pred
                    else:
                        return np.mean(scores)
                """
                st.code(body_3, language="python")

        
    with st.container(border=True):
        st.subheader("実験の概要", divider='rainbow')
        st.markdown("""
            各質問項目に対して適切な重み付けを山登り法で行う  \n
            - 各ステップで特徴量すべてに対して[-ε,0,+ε]の3方向で評価、重みの更新を行う
                - 初期値：全て1,乱数
            - 各ステップで特徴量すべてに対して[-ε,+ε]の2方向で評価、重みの更新を行う
                - 初期値：全て1,乱数 \n
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

    with st.container(border=True):
        st.subheader("実験結果", divider='rainbow')
        st.markdown("""
        - [-ε,0,+ε]の3方向
        """)
        img1 = Image.open('picture/20250613/スクリーンショット 2025-06-13 7.56.53.png')
        img2 = Image.open('picture/20250613/0.1重み初期1.png')
        img3 = Image.open('picture/20250613/1初期1_100.png')
        img4 = Image.open('picture/20250613/third0.1.png')
        img5 = Image.open('picture/20250613/third1_100.png')
        st.image(img1, caption='欠損値削除', use_container_width=True)
        # カラムを3つ作成
        col1, col2 = st.columns(2)
        # 各カラムに画像を表示
        with col1:
            st.image(img2, caption="初期値：全1,更新の大きさ：0.1", use_container_width=True)
        with col2:
            st.image(img3, caption="初期値：全1,更新の大きさ：1", use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.image(img4, caption="初期値：乱数,更新の大きさ：0.1", use_container_width=True)
        with col4:
            st.image(img5, caption="初期値：乱数,更新の大きさ：1", use_container_width=True)
        st.markdown("""
        - [-ε,+ε]の2方向
        """)
        img1 = Image.open('picture/20250613/スクリーンショット 2025-06-13 7.57.46.png')
        img2 = Image.open('picture/20250613/second全1_100_0.1.png')
        img3 = Image.open('picture/20250613/second全1.0100_1.png')
        img4 = Image.open('picture/20250613/second100_0.01.png')
        img5 = Image.open('picture/20250613/second100_1.png')
        st.image(img1, caption='欠損値削除', use_container_width=True)
        # カラムを3つ作成
        col1, col2 = st.columns(2)
        # 各カラムに画像を表示
        with col1:
            st.image(img2, caption="初期値：全1,更新の大きさ：0.1", use_container_width=True)
        with col2:
            st.image(img3, caption="初期値：全1,更新の大きさ：1", use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.image(img4, caption="初期値：乱数,更新の大きさ：0.1", use_container_width=True)
        with col4:
            st.image(img5, caption="初期値：乱数,更新の大きさ：1", use_container_width=True)

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

elif data_processing == "2025年7月11日":
    st.subheader('2025年7月11日')
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
            - 試行回数：100（更新されなかったら終了）
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
            - 質問表：PainDITACT,BS-POP,FUSION
            - 欠損値の補完：欠損値削除
            - 標準化：あり
            - 重み付けの初期値
            - 全て1.0
            - 全て乱数
                - 山登り法は初期値によって局所最適解を求めてしまう可能性があるから
            - 試行回数：100（更新されなかったら終了）
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
            - RBFカーネル（Radial Basis Function） \n
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
            - 試行回数：100（更新されなかったら終了）
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
        
        st.subheader("実験の概要④", divider='rainbow')
        st.markdown("""
        3つのクラス（侵害・神経・不明）のアンケートデータにおける各質問項目（P1〜D18）の点数ごとの回答数を棒グラフで可視化した
        - 目的：クラス（侵害・神経・不明）ごとに多く現れる回答パターンや影響の大きい特徴量を見つけるため。
        """)
        img = Image.open('picture/20250711/回答分布.png')
        st.image(img, use_container_width=True)
        st.markdown("""
        - 横軸：各質問項目に対するスコア、縦軸：回答人数
        - 青：侵害受容性疼痛、水色：神経受容性疼痛、赤：不明 \n
        → P8,P12は侵害受容性疼痛と神経障害性疼痛の患者を識別する重要な特徴量？重みを極端に変えたら結果も良くなる？
        - P8：ピリピリしたり、チクチク刺したりするような感じ（虫が歩いているような、電気が流れているような感じ）がありますか？
        - P12：痛みのある場所に、痺れを感じますか？
        """)


    with st.container(border=True):
        st.subheader("実験結果①", divider='rainbow')
        st.markdown("""
        - [前回の実験]PainDITECT・FUSIONの質問表
        """)
        img1 = Image.open('picture/20250711/PAIN.png')
        img2 = Image.open('picture/20250711/FUSION.png')
        st.image(img1, caption='PainDITECT', use_container_width=True)
        st.image(img2, caption="FUSION", use_container_width=True)
        st.markdown("""
        - 正答率は変わらなかった
        """)

        st.subheader("実験結果②", divider='rainbow')
        st.markdown("""
        - 更新幅を小さい範囲で山登り法を実施
        """)
        img1 = Image.open('picture/20250711/更新幅.png')
        st.image(img1, caption='FUSION', use_container_width=True)
        st.markdown("""
        - P11であれば重みが上がったものもあれば、下がったものがあり、判断が難しい
        """)

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
        - BS-POPの正答率で70%と今までと比較しても高く出た
        - 変更重み（BS-POP）：D1,D2,D7,D10,D11
        """)

        st.markdown("""
        - 補足
        """)
        img1 = Image.open('picture/20250711/標準化内積.png')
        st.image(img1, caption='標準化データの内積の分布', use_container_width=True)


    with st.container(border=True):
        st.subheader("今後の予定の確認", divider='rainbow')
        st.markdown("""
        - クラスごとの点数の回答数を可視化した結果や今回の実験結果からP8を中心に他の重みの初期値を変えて実験する
        - 他の欠損値補完での実験
        - パラメータの探索範囲を細かくする
        - 遺伝的アルゴリズムによる重み付け
        """)

elif data_processing == "2025年10月17日":
    st.subheader('何やってるんですか、勉強してください', divider='rainbow')
    gif_path = "picture/kounogento.gif"
    st.image(gif_path,caption="河野玄斗")