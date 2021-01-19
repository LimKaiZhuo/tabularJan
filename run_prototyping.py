import pandas as pd

from own_package.prototyping import lvl1_randomsearch, xgb_optuna
from own_package.analysis import analyze_xgb_optuna


def selector(case):
    if case == 1:
        train = pd.read_csv('./inputs/train.csv')
        test = pd.read_csv('./inputs/test.csv')
        lvl1_randomsearch(rawdf=train, testdf=test, results_dir='./results/lvl1_randomsearch_pp1_lt0',
                          pp_choice=1, lt_choice=None)
    elif case == 2:
        train = './inputs/train.csv'
        test = './inputs/test.csv'
        xgb_optuna(train_dir=train, test_dir=test, results_dir='./results/xgb_optuna_n20_pp1_lt0_s17')
    elif case == 2.1:
        analyze_xgb_optuna('./results/xgb_optuna - 2', train_dir='./inputs/train.csv', test_dir='./inputs/test.csv')

selector(2)



