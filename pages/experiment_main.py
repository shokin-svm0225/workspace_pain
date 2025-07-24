from importlib import import_module
import streamlit as st

choice = st.selectbox("どの実験？", ["poly", "rbf"])

if choice == "poly":
    module = import_module("experiment.thirdshift_experiment")  # ✅ ドット記法
    module.run_thirdshift_experiment()
elif choice == "rbf":
    module = import_module("experiment.secondshift_experiment")
    module.run_secondshift_experiment()