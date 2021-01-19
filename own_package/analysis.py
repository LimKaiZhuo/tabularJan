import pandas as pd
import pickle

from own_package.prototyping import xgb_optuna_predict


def analyze_xgb_optuna(results_dir, train_dir=None, test_dir=None):
    with open(f'{results_dir}/results_store.pkl', 'rb') as f:
        study = pickle.load(f)
    results = pd.DataFrame([{'score': x.value, **x.params} for x in study.trials]).sort_values('score')
    results.to_excel(f'{results_dir}/optuna.xlsx')

    if train_dir is not None:
        xgb_optuna_predict(best_params=study.best_params,
                           train_dir=train_dir,
                           test_dir=test_dir,
                           results_dir=results_dir,pp_choice=1,lt_choice=0,seed=17)




