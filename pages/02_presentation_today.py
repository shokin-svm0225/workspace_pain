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
    st.subheader("アジェンダ", divider='rainbow')
    st.markdown("""
    - 研究内容
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
      - PainDETECTスコア総和による判定：正答率は55.4%，感度/特異度は侵害受容性疼痛 80.1%/38.8%，神経障害性疼痛 15.3%/94.2%，原因不明 21.9%/81.9%
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