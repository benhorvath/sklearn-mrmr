"""
Main implementation for sklearn-compatible feature selection via mRMR.
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, f_regression, \
    mutual_info_classif, mutual_info_regression
from sklearn.utils import check_X_y


def calculate_mi_mrmr(
    X: pd.DataFrame, 
    y: pd.Series, 
    k: int, 
    n_jobs: int = 1,
    operation: str = 'difference',
) -> List[Tuple[np.ndarray, List, int]]:
    """
    Selects the top `k` features of a dataset using the Minimum Redundancy
    Maximum Relevance (MRMR) algorithm (original mutual information design).
    
    The MRMR algorithm selects the subset of features that are both highly
    relevant to the target variable as well as minimally redundant with
    respect to each other. This function computes both relevance and
    redundancy using mututal information. This calculation can be slow, so an
    argument exists to run it in parallel `n_jobs`. Users can choose between
    using subtraction or division to balance the trade-off between relevance 
    and redundancy. Both regression and classification tasks are supported.

    Parameters
    ----------
    X : np.ndarray
        The input samples with shape (n_samples, n_features).
    
    y : np.ndarray
        The target values with shape (n_samples,).
        
    k : int
        Number of features to select.

    n_jobs : int, default=1
        Number of jobs to calculate MI.
    
    operation : str, default='difference'
        Whether subtract or divide relevancy and redundancy.

    Returns
    -------
    mrmr_scores : np.ndarray
        The MRMR scores of selected features.

    selected_features : list
        Names of selected features.

    n_features_in : int
        Number of features in input data.
    """
    
    X, y = check_X_y(X, y)
    
    p = X.shape[1]
    
    is_classification = np.issubdtype(y.dtype, np.str_)
    
    if is_classification:
        score_func = lambda X, y: mutual_info_classif(X, y, n_jobs=n_jobs)
    else:
        score_func = lambda X, y: mutual_info_regression(X, y, n_jobs=n_jobs)
        
    relevance_scores = score_func(X, y)
    
    # Initialize containers for loop
    redundancy_scores = np.ones([p, p])
    mrmr_scores = []
    selected_features = []
    not_selected = list(range(p))
    
    for i in range(k):
    
        if i > 0:
            last_selected = selected_features[-1]
            redundancy_scores[not_selected, last_selected] = score_func(X[:, not_selected], X[:, last_selected])
    
        relevance_score = relevance_scores[not_selected]
    
        if selected_features:
            redundancy_score = np.nanmean(redundancy_scores[not_selected][:, selected_features], axis=1)
        else:
            redundancy_score = np.ones(len(not_selected))
    
        # Compare relevance and redundancy
        if operation == 'difference':
            score = relevance_score - redundancy_score
        elif operation == 'quotient':
            score = relevance_score / redundancy_score
        else:
            raise ValueError(f'Invalid method: {operation}.')
        
        best = np.argmax(score)
        best_feature = not_selected[best]
        best_score = score[best]
    
        selected_features.append(best_feature)
        not_selected.remove(best_feature)
        mrmr_scores.append(best_score)
    
    return mrmr_scores, selected_features, p

def calculate_ftest_mrmr(
    X: pd.DataFrame, 
    y: pd.Series, 
    k: int,
    operation: str = 'difference'
) -> List[Tuple[np.ndarray, List, int]]:
    """
    Selects the top `k` features of a dataset using the Minimum Redundancy Maximum Relevance 
    (MRMR) algorithm (F-test variant).

    The MRMR algorithm selects the subset of features that are both highly relevant to the target
    variable as well as minimally redundant with respect to each other. This function computes 
    relevance using the F-statistic, and redundancy based on correlation, ensuring speedy
    calculation. Users can choose between using subtraction or division to balance the trade-off
    between relevance and redundancy. This function can be used for both regression and
    classification.

    Parameters
    ----------
    X : pd.DataFrame
        The input samples with shape (n_samples, n_features).
    
    y : pd.Series
        The target values with shape (n_samples,).

    k : int
        Number of features to select.

    operation : str, default='difference'
        Whether to use subtraction or division between relevancy and redundancy.

    Returns
    -------
    mrmr_scores : np.ndarray
        The MRMR scores of selected features.

    selected_features : list
        Names of selected features.

    n_features_in : int
        Number of features in input data.
    """
    
    X, y = check_X_y(X, y)
    
    p = X.shape[1]
    
    # is_classification = np.issubdtype(y.dtype, np.str_)
    is_classification = True
    
    # Prepare relevance scores
    if is_classification:
        relevance_func = lambda X, y: f_classif(X, y)[0]
    else:
        relevance_func = lambda X, y: f_regression(X, y)[0]

    relevance_scores = relevance_func(X, y)
    
    # Initialize containers for loop
    redundancy_scores = np.ones([p, p])
    selected_features = []
    not_selected = list(range(p))
    mrmr_scores = []
    
    for i in range(k):
        
        if i > 0:
            last_selected = selected_features[-1]
            # Calculate correlation matrix of the relevant columns
            redundancy_scores[not_selected, last_selected] = np.abs(np.corrcoef(X[:, not_selected].T, X[:, last_selected].T)[:-1, -1])
            
        relevance_score = relevance_scores[not_selected]
        
        if selected_features:
               redundancy_score = np.nanmean(redundancy_scores[not_selected][:, selected_features], axis=1)
        else:
            redundancy_score = np.ones(len(not_selected))
            
        # Compare relevance and redundancy
        if operation == 'difference':
            score = relevance_score - redundancy_score
        elif operation == 'quotient':
            score = relevance_score / redundancy_score
        else:
            raise ValueError(f"Invalid method: {operation}.")
            
        best = np.argmax(score)
        best_feature = not_selected[best]
        best_score = score[best]
    
        selected_features.append(best_feature)
        not_selected.remove(best_feature)
        mrmr_scores.append(best_score)
        
    return mrmr_scores, selected_features, p

class MRMRFeatureSelector(BaseEstimator, TransformerMixin):
    """
    MRMR (Minimum Redundancy Maximum Relevance) feature selector.

    This transformer selects features based on MRMR criterion.

    Parameters
    ----------
    n_features_to_select : int, default=5
        Number of features to select.

    method : str, default='ftest'
        Method to use for MRMR feature selection. Valid choices are:
        ['mi', 'mi_quotient', 'ftest', 'ftest_quotient'].

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    Attributes
    ----------
    scores_ : array-like, shape (n_features_to_select,)
        MRMR scores of selected features.

    n_features_in_ : int
        Number of features in input dataframe.

    selected_features_ : array-like, shape (n_features_to_select,)
        Array of names of selected features.
    """

    VALID_METHODS = ['mi', 'mi_quotient', 'ftest', 'ftest_quotient']

    _parameter_constraints = {
        'n_features_to_select': [int],
        'method': VALID_METHODS,
        'n_jobs': [int]
    }

    def __init__(
        self, 
        n_features_to_select: int = 5, 
        method: str = 'ftest', 
        n_jobs: int = 1
    ) -> None:
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

        Methodology is chosen by user-specified parameters.

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

        if self.method == 'mi':
            results = calculate_mi_mrmr(X, y, k=self.n_features_to_select,
                                        n_jobs=self.n_jobs)
        elif self.method == 'mi_quotient':
            results = calculate_mi_mrmr(X, y, k=self.n_features_to_select,
                                        n_jobs=self.n_jobs, operation='quotient')
        elif self.method == 'ftest':
            results = calculate_ftest_mrmr(X, y, k=self.n_features_to_select)
        elif self.method == 'ftest_quotient':
            results = calculate_ftest_mrmr(X, y, k=self.n_features_to_select,
                                           operation='quotient')
        else:
            raise ValueError(f'Invalid method: {self.method}.')
    
        self.scores_ = results[0]
        self.selected_features_ = results[1]
        self.n_features_in_ = results[2]

        return self
    
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the input samples by selecting the features.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.

        Returns
        -------
        X_transformed : Union[pd.DataFrame, np.ndarray]
            The transformed sample. Will return as the same type of X.
        """
        if type(X) == pd.DataFrame:
            selected_columns = X.columns[self.selected_features_]
            return X[selected_columns]
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
        X_transformed : Union[pd.DataFrame, np.ndarray]
            The transformed sample. Will return as the same type of X.
        """
        self.fit(X, y)
        return self.transform(X)
    