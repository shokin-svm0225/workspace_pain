from importlib import import_module
import streamlit as st

choice = st.selectbox("どの実験？", ["山登り法3_線形カーネル", "山登り法2_線形カーネル","山登り法3_多項式カーネル", "山登り法2_多項式カーネル",
                                "山登り法3_rbfカーネル", "山登り法2_rbfカーネル", "山登り法3_シグモイドカーネル", "山登り法2_シグモイドカーネル"])

if choice == "山登り法3_線形カーネル":
    module = import_module("experiment.thirdshift_linear_experiment")  # ✅ ドット記法
    module.run_thirdshift_experiment()
elif choice == "山登り法2_線形カーネル":
    module = import_module("experiment.secondshift_linear_experiment")
    module.run_secondshift_experiment()
elif choice == "山登り法3_多項式カーネル":
    module = import_module("experiment.thirdshift_poly_experiment") 
    module.run_thirdshift_experiment()
elif choice == "山登り法2_多項式カーネル":
    module = import_module("experiment.secondshift_poly_experiment")
    module.run_secondshift_experiment()
elif choice == "山登り法3_rbfカーネル":
    module = import_module("experiment.thirdshift_rbf_experiment")
    module.run_thirdshift_experiment()
elif choice == "山登り法2_rbfカーネル":
    module = import_module("experiment.secondshift_rbf_experiment")
    module.run_secondshift_experiment()
elif choice == "山登り法3_シグモイドカーネル":
    module = import_module("experiment.thirdshift_sigmoid_experiment")
    module.run_thirdshift_experiment()
elif choice == "山登り法2_シグモイドカーネル":
    module = import_module("experiment.secondshift_sigmoid_experiment")
    module.run_secondshift_experiment()