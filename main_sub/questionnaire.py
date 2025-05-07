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
    st.title('質問表を表示')
    st.markdown('#### PainDETECT')
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/painditect.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, caption='参考文献：https://gunma-pt.com/wp-content/uploads/2015/03/paindetect.pdf', use_container_width=True)

    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/painDETECT-Q.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, caption='参考文献：https://www.researchgate.net/figure/The-painDETECT-Questionnaire-Japanese-version-PDQ-J-doi_fig3_257465057', use_container_width=True)

    st.markdown('#### BS-POP')
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/bspop_医師.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, use_container_width=True)

    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/bspop_患者.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, caption='参考文献：http://www.onitaiji.com/spine/evaluation/0.pdf', use_container_width=True)