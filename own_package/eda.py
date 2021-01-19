import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from own_package.others import plot_barh, create_results_directory


def eda_1():
    train = pd.read_csv('./inputs/train.csv')
    results_dir = create_results_directory('./results/eda/eda_1')
    plot_barh(train.isnull().sum(), title='null%', total_count=train.shape[0], plot_dir=f'{results_dir}/null_percent.png')

    train = pd.melt(train, id_vars='id', value_name='value', var_name='name')
    g = sns.FacetGrid(train, col='name', col_wrap=5, sharex=False)
    g.map(sns.kdeplot, 'value')
    plt.savefig(f'{results_dir}/kde.png', bbox_inches='tight')
    plt.close()






