from importlib import import_module
import streamlit as st

choice = st.selectbox("どの実験？", ["山登り法3_線形カーネル", "山登り法2_線形カーネル"])

if choice == "山登り法3_線形カーネル":
    module = import_module("experiment.thirdshift_linear_experiment")  # ✅ ドット記法
    module.run_thirdshift_experiment()
elif choice == "山登り法2_線形カーネル":
    module = import_module("experiment.secondshift_linear_experiment")
    module.run_secondshift_experiment()