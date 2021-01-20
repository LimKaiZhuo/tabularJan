import pandas as pd

from own_package.prototyping import lvl1_randomsearch, xgb_optuna
from own_package.analysis import analyze_xgb_optuna, average_prediction


def selector(case):
    if case == 1:
        train = pd.read_csv('./inputs/train.csv')
        test = pd.read_csv('./inputs/test.csv')
        lvl1_randomsearch(rawdf=train, testdf=test, results_dir='./results/lvl1_randomsearch_pp1_lt0',
                          pp_choice=1, lt_choice=None)
    elif case == 2:
        train = './inputs/train.csv'
        test = './inputs/test.csv'
        xgb_optuna(train_dir=train, test_dir=test, results_dir='./results/lightgbm_optuna_n200_pp1_lt0_s17')
        xgb_optuna(train_dir=train, test_dir=test, results_dir='./results/lightgbm_optuna_n200_pp1_lt0_s18')
        xgb_optuna(train_dir=train, test_dir=test, results_dir='./results/lightgbm_optuna_n200_pp1_lt0_s19')
        xgb_optuna(train_dir=train, test_dir=test, results_dir='./results/lightgbm_optuna_n200_pp1_lt0_s20')
    elif case == 2.1:
        analyze_xgb_optuna('./results/lightgbm_optuna_n10_pp1_lt0_s19', train_dir='./inputs/train.csv', test_dir='./inputs/test.csv')
    elif case == 2.2:
        results_dir_store = [f'./results/xgb_optuna_n30_pp1_lt0_s{s}' for s in [17,18,19,20]] + \
                            [f'./results/xgb_optuna_n40_pp1_lt0_s{s}' for s in range(21,33)]
        average_prediction(results_dir_store=results_dir_store)

selector(2)



