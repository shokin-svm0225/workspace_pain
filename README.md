# workspace_pain

KeyError: "['step_size'] not in index"
Traceback:
File "C:\Users\pc\Desktop\workspace_pain\pages\04_experiment_main.py", line 27, in <module>
    module.run_shift_pca_experiment()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
File "C:\Users\pc\Desktop\workspace_pain\experiment\shift_pca_random_experiment.py", line 802, in run_shift_pca_experiment
    small_df = results_df[["step_size", "kernel", "C", "gamma", "degree", "coef0", "score"]].copy()
               ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\pc\AppData\Local\Programs\Python\Python314\Lib\site-packages\pandas\core\frame.py", line 4119, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
File "C:\Users\pc\AppData\Local\Programs\Python\Python314\Lib\site-packages\pandas\core\indexes\base.py", line 6212, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\pc\AppData\Local\Programs\Python\Python314\Lib\site-packages\pandas\core\indexes\base.py", line 6264, in _raise_if_missing
    raise KeyError(f"{not_found} not in index")
