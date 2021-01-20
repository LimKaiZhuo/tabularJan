import pandas as pd
import pickle

from own_package.prototyping import xgb_optuna_predict
from own_package.others import create_results_directory


def analyze_xgb_optuna(results_dir, train_dir=None, test_dir=None):
    with open(f'{results_dir}/results_store.pkl', 'rb') as f:
        study = pickle.load(f)
    results = pd.DataFrame([{'score': x.value, **x.params} for x in study.trials]).sort_values('score')
    results.to_excel(f'{results_dir}/optuna.xlsx')

    if train_dir is not None:
        xgb_optuna_predict(best_params=study.best_params,
                           train_dir=train_dir,
                           test_dir=test_dir,
                           results_dir=results_dir)


def average_prediction(results_dir_store):
    prediction_store = [pd.read_csv(f'{results_dir}/{results_dir.split("/")[-1]}_predictions.csv')['target'] for results_dir
                        in results_dir_store]
    avg_prediction = sum(prediction_store)/len(prediction_store)

    results_dir = create_results_directory('./results/avg')
    sub = pd.DataFrame()
    sub['id'] = pd.read_csv(f'{results_dir_store[0]}/{results_dir_store[0].split("/")[-1]}_predictions.csv')['id']
    sub['target'] = avg_prediction
    sub.to_csv(f'{results_dir}/avg_predictions.csv', index=False)

    with open(f'{results_dir}/model_list.txt', "w") as output:
        output.write('\n'.join(results_dir_store))