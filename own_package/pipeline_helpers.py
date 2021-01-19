import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import scipy.optimize
import category_encoders as ce
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]


class DebuggerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, info=None):
        self.info = info

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class FinalFeatureDataframe(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        self.feature_count_ = len(self.feature_names)
        return self

    def transform(self, X):
        return pd.DataFrame(X,
                            columns=[f'{x + 1}_lvl1_' for x in
                                     range(X.shape[1] - self.feature_count_)] + self.feature_names)


class PdFunctionTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application

    Parameters
    ----------
    impute : Boolean, default False

    func : function that acts on an array of the form [n_elements, 1]
        if impute is True, functions must return a float number, otherwise
        an array of the form [n_elements, 1]

    """

    def __init__(self, func, impute=False):
        self.func = func
        self.impute = impute
        self.series = pd.Series()

    def transform(self, X, **transformparams):
        """ Transforms a DataFrame

        Parameters
        ----------
        X : DataFrame

        Returns
        ----------
        trans : pandas DataFrame
            Transformation of X
        """

        if self.impute:
            trans = pd.DataFrame(X).fillna(self.series).copy()
        else:
            trans = pd.DataFrame(X).apply(self.func).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Fixes the values to impute or does nothing

        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement

        Returns
        ----------
        self
        """
        if self.impute:
            self.series = pd.DataFrame(X).apply(self.func).squeeze()
        return self


class PdWithinGroupImputerTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application

    Parameters
    ----------
    impute : Boolean, default False

    func : function that acts on an array of the form [n_elements, 1]
        if impute is True, functions must return a float number, otherwise
        an array of the form [n_elements, 1]

    """

    def __init__(self, func, groupby, x_names):
        self.func = func
        self.groupby = groupby  # String
        self.x_names = x_names  # List of string

    def transform(self, X, **transformparams):
        X = X.copy()
        # Replace new unseen categories in groupby with Others_
        X.loc[~X[self.groupby].isin(self.groupby_categories), self.groupby] = 'Others_'
        trans = []
        for x in self.x_names:
            temp = X.copy()
            temp.loc[X[x].isnull(), x] = X.loc[X[x].isnull(), self.groupby].map(lambda n: self.df_map.loc[n, x])
            trans.append(temp[[x]])
        return pd.concat(trans, axis=1)

    def fit(self, X, y=None, **fitparams):
        # lambda x: x.value_counts().index[0]
        self.df_map = X[self.x_names + [self.groupby]].groupby(self.groupby).agg(
            {x: self.func for x in self.x_names})
        self.df_map = pd.concat([self.df_map, X[self.x_names].apply(self.func).to_frame(name='Others_').T], axis=0)
        self.groupby_categories = X[self.groupby].unique()
        return self


class PdTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type_):
        self.type_ = type_

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(self.type_)


class PdSumTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, series_name, weights=None):
        """
        series_name: name of the new summed column
        weights: weights when summing the columns together
        """
        self.series_name = series_name
        self.weights = weights

    def fit(self, X, y=None):
        if self.weights:
            assert X.shape[1] == len(self.weights)  # no. of columns = no. of weights
        return self

    def transform(self, X):
        if self.weights:
            X = X * self.weights
        return X.sum(axis=1).to_frame(self.series_name)


class Binarizer(BaseEstimator, TransformerMixin):
    def __init__(self, condition, name):
        self.condition = condition
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda x: int(self.condition(x))).to_frame(self.name)


class GroupingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *, min_count=0, min_freq=0.0, top_n=0):
        self.min_count = min_count
        self.min_freq = min_freq
        self.top_n = top_n

    def fit(self, X, y=None):
        self.group_name_ = 'Others_'
        X = X.fillna('None')  # In case there is still any nan left
        n_samples, n_features = X.shape
        counts = []
        groups = []
        other_in_keys = []
        for i in range(n_features):
            cnts = Counter(X.iloc[:, i])
            counts.append(cnts)
            if self.top_n == 0:
                self.top_n = len(cnts)
            labels_to_group = (label for rank, (label, count) in enumerate(cnts.most_common())
                               if ((count < self.min_count)
                                   or (count / n_samples < self.min_freq)
                                   or (rank >= self.top_n)
                                   )
                               )
            groups.append(np.array(sorted(set(labels_to_group))))
            other_in_keys.append(self.group_name_ in cnts.keys())
        self.counts_ = counts
        self.groups_ = groups
        self.other_in_keys_ = other_in_keys
        return self

    def transform(self, X):
        X_t = X.copy()
        X_t = X_t.fillna('None')
        _, n_features = X.shape
        for i in range(n_features):
            mask = np.isin(X_t.iloc[:, i], self.groups_[i])
            X_t.iloc[mask, i] = self.group_name_
        return X_t


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


def final_est_pipeline(feature_names, preprocess_pipeline, no_of_lvl1):
    lvl1_pred = Pipeline([
        ('create_final_df', FinalFeatureDataframe(feature_names)),
        ('lvl_1_pred', FeatureSelector([f'{x+1}_lvl1_' for x in range(no_of_lvl1)])),
    ])
    preprocess = Pipeline([
        ('create_final_df', FinalFeatureDataframe(feature_names)),
        ('final_preprocess', preprocess_pipeline),
    ])
    return FeatureUnion([
        ('lvl_1_pred', lvl1_pred),
        ('trans_features', preprocess)
    ])





