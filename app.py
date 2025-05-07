import streamlit as st
from streamlit_option_menu import option_menu
import home
import experiment
import presentation_today
import presentation_history
import dataset
import program
import data_analysis
import save_experiment
import questionnaire

with st.sidebar:
    selected = option_menu(
        menu_title="メニュー",  # サイドバータイトル
        options=['ホーム', '実験', '本日の発表', '発表履歴', 'データセット', 'プログラム', 'データ分析', '実験結果の保存', '質問表'],
        icons=["house", "play", "rocket", "rocket", "puzzle-fill", "chat-dots", "tag-fill", "chat-dots", "quora"],
        menu_icon="power",
        default_index=0,
    )
    txt = st.text_area(
        'アドバイス', height=150
    )

# 選択されたページに応じてコンテンツを表示
if selected == 'ホーム':
    home.show()

elif selected == '実験':
    experiment.show()

elif selected == '本日の発表':
    presentation_today.show()

elif selected == '発表履歴':
    presentation_history.show()

elif selected == 'データセット':
    dataset.show()

elif selected == 'プログラム':
    program.show()

elif selected == 'データ分析':
    data_analysis.show()

elif selected == '実験結果の保存':
    save_experiment.show()

else:
    questionnaire.show()