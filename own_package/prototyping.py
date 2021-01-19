import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import scipy as sp
import scipy.stats
import scipy.optimize
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from vecstack import StackingTransformer
from sklearn.metrics import make_scorer, mean_squared_error
import pickle

from own_package.own_pipeline import preprocess_pipeline_1
from own_package.others import create_results_directory


def pp_selector(preprocess_pipeline_choice, rawdf=None):
    if preprocess_pipeline_choice == 1:
        preprocess_pipeline = preprocess_pipeline_1(rawdf)
    return preprocess_pipeline


def lvl1_randomsearch(rawdf, testdf, results_dir, pp_choice, lt_choice=None, n_iter=100):
    '''

    :param rawdf:
    :param results_dir:
    :param pp_choice: preprocessing choice
    :param lt_choice: label tranformation choice. None is no transformation.
    :return:
    '''
    results_dir = create_results_directory(results_dir)
    x_train = rawdf.iloc[:, :-1]
    y_train = rawdf.iloc[:, -1]
    x_test = testdf
    model_store = ['svr']
    model_iter = {
        'xgb': 10,
        'rf': 10,
        'et': 10,
        'eecv': 5,
        'svr': 10
    }
    model_object = {
        'xgb': XGBRegressor(),
        'rf': RandomForestRegressor(),
        'et': ExtraTreesRegressor(),
        'eecv': ElasticNetCV(),
        'svr': SVR()
    }
    model_param = {
        'xgb': {'xgb__n_estimators': scipy.stats.randint(150, 1000),
                'xgb__learning_rate': scipy.stats.uniform(0.01, 0.59),
                'xgb__subsample': scipy.stats.uniform(0.3, 0.6),
                'xgb__max_depth': scipy.stats.randint(1, 16),
                'xgb__colsample_bytree': scipy.stats.uniform(0.5, 0.4),
                'xgb__gamma': scipy.stats.expon(scale=0.01),
                },
        'rf': {"rf__max_depth": [None],
               "rf__max_features": scipy.stats.randint(1, 11),
               "rf__min_samples_split": scipy.stats.randint(2, 11),
               "rf__min_samples_leaf": scipy.stats.randint(1, 11),
               "rf__bootstrap": [False],
               "rf__n_estimators": scipy.stats.randint(10, 300), },
        'et': {"et__max_depth": [None],
               "et__max_features": scipy.stats.randint(1, 11),
               "et__min_samples_split": scipy.stats.randint(2, 11),
               "et__min_samples_leaf": scipy.stats.randint(1, 11),
               "et__bootstrap": [False],
               "et__n_estimators": scipy.stats.randint(10, 300), },
        'eecv': {'eecv__l1_ratio': scipy.stats.uniform(0, 1)},
        'svr': {'svr__kernel': ['linear', 'rbf', 'sigmoid'],
                "svr__C": scipy.stats.expon(scale=.01),
                "svr__gamma": scipy.stats.expon(scale=.01), }
    }
    results_store = {}

    preprocess_pipeline = pp_selector(pp_choice, rawdf)

    if lt_choice is None:
        scorer = 'neg_root_mean_squared_error'
    elif lt_choice == 1 or lt_choice == 2:
        y_train = np.log(y_train)
        scorer = 'neg_root_mean_squared_error'

    for model_name in model_store:
        model = Pipeline([
            ('preprocess', preprocess_pipeline),
            (model_name, model_object[model_name])
        ])

        clf = RandomizedSearchCV(model,
                                 param_distributions=model_param[model_name],
                                 cv=5,
                                 n_iter=model_iter[model_name],
                                 scoring=scorer,
                                 verbose=1,
                                 n_jobs=-1, refit=True)

        clf.fit(x_train, y_train)
        results_store[model_name] = clf.cv_results_

        if lt_choice is None:
            pred_y_test = clf.predict(x_test)
        elif lt_choice == 1:
            pred_y_test = np.exp(clf.predict(x_test))

        sub = pd.DataFrame()
        sub['id'] = x_test['id']
        sub['target'] = pred_y_test
        sub.to_csv(f'{results_dir}/{model_name}_{results_dir.split("/")[-1]}_predictions.csv', index=False)

    with open(f'{results_dir}/results_store.pkl', 'wb') as f:
        pickle.dump(results_store, f)


def xgb_optuna(train_dir, test_dir, results_dir):
    meta_params = results_dir.split('_')
    n_iter = int(meta_params[-4][1:])
    pp_choice = int(meta_params[-3][2:])
    lt_choice = int(meta_params[-2][2:])
    seed = int(meta_params[-1].split(' - ')[0][1:])

    train_df = pd.read_csv(train_dir)
    x_main = train_df.iloc[:, :-1]
    y_main = train_df.iloc[:, -1]
    pp = pp_selector(pp_choice, x_main)

    # val2 is used for early stopping and will not be touched again
    x, x_val2, y, y_val2 = train_test_split(x_main, y_main, test_size=0.1, random_state=seed)

    def objective(trial):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=seed)
        pp.fit(x_train)
        x_train = pp.transform(x_train)
        x_val = pp.transform(x_val)
        params = {  # 'tree_method': 'gpu_hist',
            'random_state': np.random.RandomState(seed),
            'verbose': 0,
            'n_estimators': 3000,
            'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.5),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1),
            'max_depth': trial.suggest_int('max_depth', 1, 15),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1),
            'gamma': trial.suggest_loguniform('gamma', 1e-6, 5),
        }
        regr = XGBRegressor(**params)
        regr.fit(x_train, y_train,
                 eval_metric='rmse',
                 eval_set=[(pp.transform(x_val2), y_val2)],
                 early_stopping_rounds=10)
        y_val_pred = regr.predict(x_val)
        return np.sqrt(mean_squared_error(y_val, y_val_pred))

    study = optuna.create_study(study_name='xgb_optuna', direction='minimize')
    study.optimize(objective, n_trials=n_iter)
    results_dir = create_results_directory(results_dir)
    with open(f'{results_dir}/results_store.pkl', 'wb') as f:
        pickle.dump(study, f)

    xgb_optuna_predict(best_params=study.best_params, train_dir=train_dir, test_dir=test_dir, results_dir=results_dir)


def xgb_optuna_predict(best_params, train_dir, test_dir, results_dir):
    meta_params = results_dir.split('_')
    n_iter = int(meta_params[-4][1:])
    pp_choice = int(meta_params[-3][2:])
    lt_choice = int(meta_params[-2][2:])
    seed = int(meta_params[-1].split(' - ')[0][1:])

    train_df = pd.read_csv(train_dir)
    x_main = train_df.iloc[:, :-1]
    y_main = train_df.iloc[:, -1]
    pp = pp_selector(pp_choice, x_main)

    # val2 is used for early stopping and will not be touched again
    x_train, x_val2, y_train, y_val2 = train_test_split(x_main, y_main, test_size=0.1, random_state=seed)
    pp.fit(x_train)
    x_train = pp.transform(x_train)
    params = {'random_state': np.random.RandomState(seed),
              'verbose': 0,
              'n_estimators': 3000,
              **best_params}
    regr = XGBRegressor(**params)
    regr.fit(x_train, y_train,
             eval_metric='rmse',
             eval_set=[(x_train, y_train), (pp.transform(x_val2), y_val2)],
             early_stopping_rounds=10)

    del train_df
    x_test = pd.read_csv(test_dir)
    y_test_pred = regr.predict(pp.transform(x_test))
    sub = pd.DataFrame()
    sub['id'] = x_test['id']
    sub['target'] = y_test_pred
    sub.to_csv(f'{results_dir}/{results_dir.split("/")[-1]}_predictions.csv', index=False)