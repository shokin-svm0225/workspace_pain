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
    - 実験1
    - 実験2
    - 今後の予定の確認
    - アドバイス
    """)

with st.container(border=True):
    st.subheader("実験1", divider='rainbow')
    st.markdown("""
    前回まで行った実験(主成分分析＋山登り法＋ランダムジャンプ)について、RBFカーネル・シグモイドカーネルでの実験を行ったので、その結果の説明と4つのカーネルでの比較を行う。
    
    **実験全体の流れ（主成分分析＋山登り法＋ランダムジャンプ）**\n
    1. データの前処理
      - 標準化：平均0、分散1に変換
      - PCA：次元削減
        - 主成分数(80%)： 7次元(PainDETECT)、11次元(BS-POP)、18次元(FUSION)
      - 初期重み：各PCA特徴量に対する重みを設定（全て1.0とする）
    2. 特徴量重みの最適化
      - PCAの各主成分に対する重みを山登り法＋ランダムジャンプで最適化
      - グリッドサーチによるパラメータチューニング
        - C：0.1,1
        - gamma：3,5,8
        - coef0：0.01
        - degree：2,3
    3. 2,3を設定した回数だけ繰り返し、最適モデルを探索
      - 反復回数：1000
      - 2での判定性能は5-分割交差検証で評価
    4 最終評価
      - 最もスコアの高いモデル（重み・パラメータ）を保存
      - 山登り法によるスコアの変化を可視化
                
    **山登り法＋ランダムジャンプ（特徴量の重み）**\n
    1. 近傍探索: 現在のパラメータから「少し増やす」「少し減らす」を試し、スコアが上がるなら移動
    - Step Size：0.01 ※加算
    - 反復回数：1000
    2. スコアの停滞でランダムジャンプを発動
    - スコアの停滞が10回で大ジャンプ、20回で小ジャンプ
    - 全特徴量（PCAの数）のうち、小ジャンプ(10%) / 大ジャンプ(20%)の重みを強制的に変更
    - 強度：小ジャンプ[0.85,1.15](±15%) / 大ジャンプ[0.5,1.5](±50%)

    **実験結果**\n
    - 線形カーネル、多項式カーネル同様にシグモイドカーネル、RBFカーネルでも主成分分析・ランダムジャンプを導入することで、次元削減しても高精度を維持
    - 主成分分析により計算時間が約30%〜60%短縮
    - シグモイドカーネルに関しては、他のカーネルと比較してもスコアが低く、パラメータのチューニングがうまくいっていない可能性がある
      - シグモイドカーネルのSVMはニューラルネットワークと同じ動きをし、定式的に少しずれると決定境界が大きく変わってしまうため、最適なパラメータを定めるのが困難
    """)
    # タブの作成
    tab1, tab2 = st.tabs(["シグモイド&RBFカーネル", "線形&多項式カーネル"])
    # 各タブに内容を追加
    with tab1:
      img1 = Image.open('picture/20251219/rbf_oldexperiment.png')
      st.image(img1, caption="RBFカーネル", width="stretch")
      img2 = Image.open('picture/20251219/sigmoid_oldexperiment.png')
      st.image(img2, caption="シグモイドカーネル", width="stretch")

    with tab2:
      img3 = Image.open('picture/20251219/linear_oldexperiment.png')
      st.image(img3, caption="線形カーネル", width="stretch")
      img4 = Image.open('picture/20251219/poly_oldexperiment.png')
      st.image(img4, caption="多項式カーネル", width="stretch")
        
with st.container(border=True):
    st.subheader("実験2", divider='rainbow')
    st.markdown("""
    実験1では、山登り法＋ランダムジャンプで質問項目に対する重みを探索していたが、
    実験2ではSVMのハイパーパラメータを山登り法＋ランダムジャンプで探索し、その結果を用いて質問項目に対する重みを探索することを繰り返してスコアの向上を図る。 \n
    
    **実験全体の流れ（交互最適化）**\n
    1. データの前処理
      - 標準化：平均0、分散1に変換
      - PCA：次元削減
        - 主成分数(80%)： 7次元(PainDETECT)、11次元(BS-POP)、18次元(FUSION)
      - 初期重み：各PCA特徴量に対する重みを設定（全て1.0とする）
    2. SVMパラメータの最適化
      - 現在の「特徴量の重み」を固定した状態で、SVMのパラメータ（C, gamma, coef0, degree）を山登り法＋ランダムジャンプで最適化
      - 問題：C,gammaの最適化の候補範囲が広く、数を一つに絞って探索するのは非効率的
        - 並列化により複数の初期値（C, gamma, coef0）から探索する
        - degreeは定式的に2,3が最適だと考えられるので、総当たりで割り当てる\n
    3. 特徴量重みの最適化
      - 1で見つけた最適なSVMパラメータを固定した状態で、PCAの各主成分に対する重みを山登り法＋ランダムジャンプで最適化
      - ジャンプさせる特徴量：ランダム(小ジャンプ：特徴量数の10%を選択,大ジャンプ：特徴量数の20%を選択)\n
    4. 2,3を設定した回数だけ繰り返し、最適モデルを探索
      - サイクル回数：3
      - 2,3での判定性能は5-分割交差検証で評価
    5 最終評価
      - 最もスコアの高いモデル（重み・パラメータ）を保存
      - 2,3間のスコアの変化を可視化
                
    **山登り法＋ランダムジャンプ（特徴量の重み）**\n
    1. 近傍探索: 現在のパラメータから「少し増やす」「少し減らす」を試し、スコアが上がるなら移動
    - Step Size：0.01 ※加算
    - 反復回数：1000
    2. スコアの停滞でランダムジャンプを発動
    - スコアの停滞が10回で大ジャンプ、20回で小ジャンプ
    - 全特徴量（PCAの数）のうち、小ジャンプ(10%) / 大ジャンプ(20%)の重みを強制的に変更
    - 強度：小ジャンプ[0.85,1.15](±15%) / 大ジャンプ[0.5,1.5](±50%)
                
    **山登り法＋ランダムジャンプ（SVMのハイパーパラメータ）**\n
    1. 近傍探索: 現在のパラメータから「少し増やす」「少し減らす」を試し、スコアが上がるなら移動
    - Step Size (C, gamma, coef0) ※加算
    - 山登りやジャンプで極端な値にならないようにパラメータ有効範囲を設ける
      - C：(min,max) = (0.01,100)
      - gamma：(min,max) = (0.01,1)
      - coef0：(min,max) = (1,10)
    2. スコアの停滞でランダムジャンプを発動
    - スコアの停滞が10回で大ジャンプ、20回で小ジャンプ
    - 全特徴量（PCAの数）のうち、全ての重みを強制的に変更
    - 強度：設定した有効範囲（Max - Min）に対して、小ジャンプ(10%) / 大ジャンプ(20%)の割合(%)だけ値をずらす
    - 例: Cの範囲が 0～100 で、小ジャンプがなら、現在の値から ±10 程度の範囲でランダムに移動させる

    **実験結果**\n
    以前まで作成した特徴量重みの山登り法のプログラムと今回作成したSVMハイパーパラメータの山登り法のプログラムの実装を分けて行った際に、
    step2からstep3まではスコアの向上が見られたが、step4で繰り返す際にスコアの低くなっているのが見られた。
    （Cycle2のスタート地点（初期値）はCycle1のベスト状態と全く同じなので、スコアは「同じ」か「上がる」はず）\n
    原因：質問項目の重みを正規化しており、本来小数点以下も何桁とあるが、自身の実装上表示では少数第2位までしか表示していなかった\n
    下の表では、step1からstep3までの結果をまとめている。
    - パラメータを広範囲で探すことで、以前よりもスコアが少し向上しているところが多い
    - サイクルが1回しか回せていないため、このスコアから上がる可能性があるのかわからない
    """)

    img1 = Image.open('picture/20251219/cross_score.png')
    st.image(img1, caption="step2,3でのスコア変化", width="stretch")


    st.markdown("""
        - step1からstep5まで一貫して実装できるプログラムを作成
          - step3での重みをそのまま次のstepに渡せるように修正
        - 線形カーネルのみで交互最適化を3サイクル回した結果が以下の通りである
          - bspopは3サイクルで少しずつスコアが向上したものの、paindetect・FUSIONはサイクル1でスコアが停滞してしまった
          - 以前までの主成分分析＋山登り法（ランダムジャンプ）の実験結果と比較すると、paindetectが72.22% → 74.07%、bspopが64.26% → 65.93%、FUSIONが73.52% → 73.33%に変化
        """)
    # タブの作成
    tab1, tab2, tab3 = st.tabs(["PainDETECT", "BS-POP", "FUSION"])
    # 各タブに内容を追加
    with tab1:
      img1 = Image.open('picture/20251219/paindetect_change.png')
      st.image(img1, caption="3サイクルでのスコア変化", width="stretch")
      csv_file_path = 'picture/20251219/paindetect_3.csv'  # ファイルパスを指定
      df = pd.read_csv(csv_file_path)
      st.dataframe(df)
      img2 = Image.open('picture/20251219/paindetect_3.png')
      st.image(img2, caption="step間のスコア変動", width="stretch")

    with tab2:
      img3 = Image.open('picture/20251219/bspop_change.png')
      st.image(img3, caption="3サイクルでのスコア変化", width="stretch")
      csv_file_path = 'picture/20251219/bspop_3.csv'  # ファイルパスを指定
      df = pd.read_csv(csv_file_path)
      st.dataframe(df)
      img4 = Image.open('picture/20251219/bspop_3.png')
      st.image(img4, caption="step間のスコア変動", width="stretch")

    with tab3:
      img5 = Image.open('picture/20251219/fusion_change.png')
      st.image(img5, caption="3サイクルでのスコア変化", width="stretch")
      csv_file_path = 'picture/20251219/fusion_3.csv'  # ファイルパスを指定
      df = pd.read_csv(csv_file_path)
      st.dataframe(df)
      img6 = Image.open('picture/20251219/fusion_3.png')
      st.image(img6, caption="step間のスコア変動", width="stretch")

with st.container(border=True):
    st.subheader("今後の予定の確認", divider='rainbow')
    st.markdown("""
    - 線形以外のカーネルで交互最適化の実験を行う
    - 進化計算での山登り法の実装
    - 他の欠損値補完での実験
    - 教師なし学習の実装
      - som（クラスタリング）
    - 決定木やるならlightgbm
    """)