from importlib import import_module
import streamlit as st

st.title('実験')

st.sidebar.header("実行方法の選択")
howto = st.sidebar.radio("実行方法", ["デフォルト", "山登り法"], index=0)

choice = st.sidebar.radio("PCAの有無", ["有", "無"], index=1)

if howto == "デフォルト" and choice == "有":
    module = import_module("experiment.shift_experiment")  # ✅ ドット記法
    module.run_shift_experiment()
elif howto == "デフォルト" and choice == "無":
    module = import_module("experiment.experiment_default")
    module.default_experiment()
elif howto == "山登り法" and choice == "有":
    module = import_module("experiment.thirdshift_poly_experiment") 
    module.run_thirdshift_experiment()
elif howto == "山登り法" and choice == "無":
    module = import_module("experiment.shift_experiment")  # ✅ ドット記法
    module.run_shift_experiment()