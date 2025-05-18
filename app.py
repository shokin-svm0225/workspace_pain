import streamlit as st
from streamlit_option_menu import option_menu

st.title('streamlit')

st.subheader('streamlitのインストール', divider='rainbow')
# コード表示
code = '''pip install streamlit'''
st.code(code, language='python')

st.subheader('プログラムの簡単な例', divider='rainbow')
# コード表示
code = '''import streamlit as st
        st.title('こんにちは、Streamlit！')
        st.write('これが私の最初のStreamlitアプリです。')'''
st.code(code, language='python')

st.subheader('実行コード', divider='rainbow')
# コード表示
code = '''streamlit run app.py'''
st.code(code, language='python')

st.subheader('参考サイト', divider='rainbow')
st.write("- [Streamlit_documentation](https://docs.streamlit.io/): streamlitのドキュメント")