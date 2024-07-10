"""
Main implementation for slearn-compatible feature selection via mRMR.
"""

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif, \
    mutual_info_regression, f_classif, f_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y


def calculate_mi_mrmr(X: np.ndarray, y: np.ndarray, n_jobs: int = 1, 
                      operation: str = 'difference', **kwargs) -> np.ndarray:
    """
    Calculate MRMR (Minimum Redundancy Maximum Relevance) scores for feature selection.

    Parameters
    ----------
    X : np.ndarray
        The input samples with shape (n_samples, n_features).
    
    y : np.ndarray
        The target values with shape (n_samples,).
    
    n_jobs : int, default=1
        Number of jobs to run in parallel.
    
    operation : str, default='difference'
        The operation to combine relevance and redundancy. Valid options are 'difference' or 'quotient'.
    
    kwargs : dict
        Additional parameters to pass to the mutual information functions.

    Returns
    -------
    mrmr_scores : np.ndarray
        The MRMR scores for each feature.
    """

    def compute_redundancy(i, j):
        """ Helper function to clean up parallel code."""
        return score_func(X[:, [i]], X[:, [j]].flatten())

    is_classification = np.issubdtype(y.dtype, np.str_)
    
    if is_classification:
        score_func = lambda X, y: mutual_info_classif(X, y, n_jobs=n_jobs, **kwargs)
    else:
        score_func = lambda X, y: mutual_info_regression(X, y, n_jobs=n_jobs, **kwargs)

    feature_indices = list(range(X.shape[1]))

    mrmr_scores = np.zeros(len(feature_indices))

    for i in feature_indices:

        other_feature_indices = feature_indices.copy()
        other_feature_indices.remove(i)

        relevance = score_func(X[:, [i]], y)

        redundancy = Parallel(n_jobs=n_jobs)(delayed(compute_redundancy)(i, j) for j in other_feature_indices)
        redundancy = np.mean(redundancy)

        if operation == 'difference':
            mrmr_score = relevance - redundancy
        else:
            mrmr_score = relevance / redundancy
        
        mrmr_scores[i] = mrmr_score

    return mrmr_scores

def calculate_ftest_mrmr(X: np.ndarray, y: np.ndarray, operation: str = 'difference') -> np.ndarray:
    """
    Short description

    Parameters
    ----------
    X : np.ndarray
        The input samples with shape (n_samples, n_features).
    
    y : np.ndarray
        The target values with shape (n_samples,).
    
    operation : str, default='difference'
        The operation to combine relevance and redundancy. Valid options are 'difference' or 'quotient'.

    Returns
    -------
    mrmr_scores : np.ndarray
        The MRMR scores for each feature.
    """

    is_classification = np.issubdtype(y.dtype, np.str_)
        
    if is_classification:
        relevance_func = lambda X, y: f_classif(X, y)[0]
    else:
        relevance_func = lambda X, y: f_regression(X, y)[0]
    
    feature_indices = list(range(X.shape[1]))
    
    redundancy_func = lambda X, y: stats.pearsonr(X, y)[0]  # TODO: kwargs, requires one hot encoded

    mrmr_scores = np.zeros(len(feature_indices))

    for i in feature_indices:

        other_feature_indices = feature_indices.copy()
        other_feature_indices.remove(i)

        relevance = relevance_func(X[:, [i]], y)

        redundancy = []
        for j in other_feature_indices:
            redundancy_ij = redundancy_func(X[:, i], X[:, [j]].flatten())
            redundancy.append(redundancy_ij)
        redundancy = np.mean(redundancy)

        if operation == 'difference':
            mrmr_score = relevance - redundancy
        else:
            mrmr_score = relevance / redundancy
        
        mrmr_scores[[i]] = mrmr_score

    return mrmr_scores

# TODO: Need to pass kwargs to here -- probably pass n_jobs through kwargs too
class MRMRFeatureSelector(BaseEstimator, TransformerMixin):
    """
    MRMR (Minimum Redundancy Maximum Relevance) feature selector.

    This transformer selects features based on MRMR criterion.

    Parameters
    ----------
    n_features_to_select : int, default=10
        Number of features to select.

    method : str, default='mi'
        Method to use for MRMR feature selection. Valid choices are:
        ['mi', 'mi_quotient', 'ftest', 'ftest_quotient'].

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    Attributes
    ----------
    scores_ : array-like, shape (n_features,)
        Scores of features.

    n_features_in_ : int
        Number of features seen during `fit`.

    selected_features_ : array-like, shape (n_features_to_select,)
        Boolean mask of selected features.
    """

    VALID_METHODS = ['mi', 'mi_quotient', 'ftest', 'ftest_quotient']

    _parameter_constraints = {
        'n_features_to_select': [int],
        'method': VALID_METHODS,
        'n_jobs': [int]
    }

    def __init__(self, 
                 n_features_to_select: int = 10, 
                 method: str = 'mi', 
                 n_jobs: int = 1) -> None:
        self.n_features_to_select = n_features_to_select
        self.method = method
        self.n_jobs = n_jobs
        self.scores_ = None
        self.n_features_in_ = None
        self.selected_features_ = None

        if self.method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method: {self.method}. Valid choices are: {self.VALID_METHODS}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MRMRFeatureSelector':
        """
        Fit the MRMR feature selector.

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.

        y : pd.Series
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y = check_X_y(X, y)

        if self.method == 'mi':
            score_func = lambda X, y: calculate_mi_mrmr(X, y, n_jobs=self.n_jobs)
        elif self.method == 'mi_quotient':
            score_func = lambda X, y: calculate_mi_mrmr(X, y, n_jobs=self.n_jobs, operation='quotient')
        elif self.method == 'ftest':
            score_func = lambda X, y: calculate_ftest_mrmr(X, y)
        elif self.method == 'ftest_quotient':
            score_func = lambda X, y: calculate_ftest_mrmr(X, y, operation='quotient')
        else:
            raise ValueError(f'Unsupported method: {self.method}')

        selector = SelectKBest(score_func, k=self.n_features_to_select)
        selector.fit(X, y)

        self.scores_ = selector.scores_
        self.n_features_in_ = selector.n_features_in_
        self.selected_features_ = selector.get_support()

        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the input samples by selecting the features.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.

        Returns
        -------
        X_transformed : np.ndarray
            The transformed samples.
        """
        if type(X) == pd.DataFrame:
            return X.loc[:, self.selected_features_]
        else:
            return X[:, self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.

        y : pd.Series
            The target values.

        Returns
        -------
        X_transformed : np.ndarray
            The transformed samples.
        """
        self.fit(X, y)
        return self.transform(X)
    