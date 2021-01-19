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

from own_package.pipeline_helpers import FeatureSelector


def preprocess_pipeline_1(rawdf):
    # Ratio or Interval pipeline: Numeric pipeline
    numeric_columns = ['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7',
       'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14']
    p_numeric = Pipeline([
        ('sel_numeric', FeatureSelector(numeric_columns)),
    ])

    return p_numeric


