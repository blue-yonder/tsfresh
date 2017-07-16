# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This module contains the filtering process for the extracted features. The filtering procedure can also be used on
other features that are not based on time series.
"""

from __future__ import absolute_import

import logging
from functools import reduce
import pandas as pd
import numpy as np
from tsfresh import defaults
from tsfresh.utilities.dataframe_functions import check_for_nans_in_columns
from tsfresh.feature_selection.feature_selector import check_fs_sig_bh


_logger = logging.getLogger(__name__)


def select_features(X, y, test_for_binary_target_binary_feature=defaults.TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
                    test_for_binary_target_real_feature=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE,
                    test_for_real_target_binary_feature=defaults.TEST_FOR_REAL_TARGET_BINARY_FEATURE,
                    test_for_real_target_real_feature=defaults.TEST_FOR_REAL_TARGET_REAL_FEATURE,
                    fdr_level=defaults.FDR_LEVEL, hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT,
                    n_jobs=defaults.N_PROCESSES, chunksize=defaults.CHUNKSIZE,
                    ml_task='auto'):
    """
    Check the significance of all features (columns) of feature matrix X and return a possibly reduced feature matrix
    only containing relevant features.

    The feature matrix must be a pandas.DataFrame in the format:

        +-------+-----------+-----------+-----+-----------+
        | index | feature_1 | feature_2 | ... | feature_N |
        +=======+===========+===========+=====+===========+
        | A     | ...       | ...       | ... | ...       |
        +-------+-----------+-----------+-----+-----------+
        | B     | ...       | ...       | ... | ...       |
        +-------+-----------+-----------+-----+-----------+
        | ...   | ...       | ...       | ... | ...       |
        +-------+-----------+-----------+-----+-----------+
        | ...   | ...       | ...       | ... | ...       |
        +-------+-----------+-----------+-----+-----------+
        | ...   | ...       | ...       | ... | ...       |
        +-------+-----------+-----------+-----+-----------+


    Each column will be handled as a feature and tested for its significance to the target.

    The target vector must be a pandas.Series or numpy.array in the form

        +-------+--------+
        | index | target |
        +=======+========+
        | A     | ...    |
        +-------+--------+
        | B     | ...    |
        +-------+--------+
        | .     | ...    |
        +-------+--------+
        | .     | ...    |
        +-------+--------+

    and must contain all id's that are in the feature matrix. If y is a numpy.array without index, it is assumed
    that y has the same order and length than X and the rows correspond to each other.

    Examples
    ========

    >>> from tsfresh.examples import load_robot_execution_failures
    >>> from tsfresh import extract_features, select_features
    >>> df, y = load_robot_execution_failures()
    >>> X_extracted = extract_features(df, column_id='id', column_sort='time')
    >>> X_selected = select_features(X_extracted, y)

    :param X: Feature matrix in the format mentioned before which will be reduced to only the relevant features.
              It can contain both binary or real-valued features at the same time.
    :type X: pandas.DataFrame

    :param y: Target vector which is needed to test which features are relevant. Can be binary or real-valued.
    :type y: pandas.Series or numpy.ndarray

    :param test_for_binary_target_binary_feature: Which test to be used for binary target, binary feature (currently unused)
    :type test_for_binary_target_binary_feature: str

    :param test_for_binary_target_real_feature: Which test to be used for binary target, real feature
    :type test_for_binary_target_real_feature: str

    :param test_for_real_target_binary_feature: Which test to be used for real target, binary feature (currently unused)
    :type test_for_real_target_binary_feature: str

    :param test_for_real_target_real_feature: Which test to be used for real target, real feature (currently unused)
    :type test_for_real_target_real_feature: str

    :param fdr_level: The FDR level that should be respected, this is the theoretical expected percentage of irrelevant
                      features among all created features.
    :type fdr_level: float

    :param hypotheses_independent: Can the significance of the features be assumed to be independent?
                                   Normally, this should be set to False as the features are never
                                   independent (e.g. mean and median)
    :type hypotheses_independent: bool

    :param n_jobs: Number of processes to use during the p-value calculation
    :type n_jobs: int

    :param chunksize: Size of the chunks submitted to the worker processes
    :type chunksize: int

    :param ml_task: The intended machine learning task. Either `'classification'`, `'regression'` or `'auto'`.
                    Defaults to `'auto'`, meaning the intended task is inferred from `y`.
                    If `y` has a boolean, integer or object dtype, the task is assumend to be classification,
                    else regression.
    :type ml_task: str

    :return: The same DataFrame as X, but possibly with reduced number of columns ( = features).
    :rtype: pandas.DataFrame

    :raises: ``ValueError`` when the target vector does not fit to the feature matrix
             or `ml_task` is not one of `'auto'`, `'classification'` or `'regression'`.
    """
    check_for_nans_in_columns(X)

    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("The type of target vector y must be one of: pandas.Series, numpy.ndarray")

    if len(X) < 2:
        raise ValueError("X must contain at least two samples.")
    elif isinstance(y, pd.Series) and not X.index.isin(y.index).all():
        raise ValueError("Index of X must be a subset of y's index")
    elif isinstance(y, np.ndarray):
        if not len(y) >= len(X):
            raise ValueError("Target vector y is shorter than feature matrix X")

        y = pd.Series(y, index=X.index)

    relevance_table = get_relevance_table(
        X, y, ml_task=ml_task, n_jobs=n_jobs, chunksize=chunksize,
        test_for_binary_target_real_feature=test_for_binary_target_real_feature,
        fdr_level=fdr_level, hypotheses_independent=hypotheses_independent,
    )

    relevant_features = relevance_table[relevance_table.relevant].Feature

    return X.loc[:, relevant_features]


def infer_ml_task(y):
    """
    Infer the machine learning task to select for.
    The result will be either `'regression'` or `'classification'`.
    If the target vector only consists of integer typed values or objects, we assume the task is `'classification'`.
    Else `'regression'`.

    :param y: The target vector y.
    :type y: pandas.Series
    :return: 'classification' or 'regression'
    :rtype: str
    """
    if y.dtype.kind in np.typecodes['AllInteger'] or y.dtype == np.object:
        ml_task = 'classification'
    else:
        ml_task = 'regression'

    _logger.warning('Infered {} as machine learning task'.format(ml_task))
    return ml_task


def get_relevance_table(X, y, ml_task='auto', test_for_binary_target_binary_feature=defaults.TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
                        test_for_binary_target_real_feature=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE,
                        test_for_real_target_binary_feature=defaults.TEST_FOR_REAL_TARGET_BINARY_FEATURE,
                        test_for_real_target_real_feature=defaults.TEST_FOR_REAL_TARGET_REAL_FEATURE,
                        fdr_level=defaults.FDR_LEVEL, hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT,
                        n_jobs=defaults.N_PROCESSES, chunksize=defaults.CHUNKSIZE):
    """
    Get the relevance table for the features contained in feature matrix `X` with respect to target vector `y`.
    The relevance table is calculated for the intended machine learning task `ml_task`.

    :param X: Feature matrix in the format mentioned before which will be reduced to only the relevant features.
              It can contain both binary or real-valued features at the same time.
    :type X: pandas.DataFrame

    :param y: Target vector which is needed to test which features are relevant. Can be binary or real-valued.
    :type y: pandas.Series or numpy.ndarray

    :param ml_task: The intended machine learning task. Either `'classification'`, `'regression'` or `'auto'`.
                    Defaults to `'auto'`, meaning the intended task is inferred from `y`.
                    If `y` has a boolean, integer or object dtype, the task is assumend to be classification,
                    else regression.
    :type ml_task: str

    :param test_for_binary_target_binary_feature: Which test to be used for binary target, binary feature (currently unused)
    :type test_for_binary_target_binary_feature: str

    :param test_for_binary_target_real_feature: Which test to be used for binary target, real feature
    :type test_for_binary_target_real_feature: str

    :param test_for_real_target_binary_feature: Which test to be used for real target, binary feature (currently unused)
    :type test_for_real_target_binary_feature: str

    :param test_for_real_target_real_feature: Which test to be used for real target, real feature (currently unused)
    :type test_for_real_target_real_feature: str

    :param fdr_level: The FDR level that should be respected, this is the theoretical expected percentage of irrelevant
                      features among all created features.
    :type fdr_level: float

    :param hypotheses_independent: Can the significance of the features be assumed to be independent?
                                   Normally, this should be set to False as the features are never
                                   independent (e.g. mean and median)
    :type hypotheses_independent: bool

    :param n_jobs: Number of processes to use during the p-value calculation
    :type n_jobs: int

    :param chunksize: Size of the chunks submitted to the worker processes
    :type chunksize: int

    :return: A pandas.DataFrame with each column of the input DataFrame X as index with information on the significance
             of this particular feature. The DataFrame has the columns
             "Feature",
             "type" (binary, real or const),
             "p_value" (the significance of this feature as a p-value, lower means more significant)
             "relevant" (True if the Benjamini Hochberg procedure rejected the null hypothesis for this feature)
    :rtype: pandas.DataFrame
    """
    if ml_task not in ['auto', 'classification', 'regression']:
        raise ValueError('ml_task must be one of: \'auto\', \'classification\', \'regression\'')
    elif ml_task == 'auto':
        ml_task = infer_ml_task(y)

    if ml_task == 'classification':
        relevance_tables = []
        for label in y.unique():
            y_label = (y == label)
            relevance_table = check_fs_sig_bh(
                X, y_label, target_is_binary=True, n_jobs=n_jobs, chunksize=chunksize,
                test_for_binary_target_real_feature=test_for_binary_target_real_feature,
                fdr_level=fdr_level, hypotheses_independent=hypotheses_independent,
            )
            relevance_tables.append((label, relevance_table))
        relevance_table = combine_relevance_tables(relevance_tables)
    elif ml_task == 'regression':
        relevance_table = check_fs_sig_bh(
            X, y, target_is_binary=False, n_jobs=n_jobs, chunksize=chunksize,
            test_for_binary_target_real_feature=test_for_binary_target_real_feature,
            fdr_level=fdr_level, hypotheses_independent=hypotheses_independent,
        )

    return relevance_table


def combine_relevance_tables(relevance_tables_with_label):
    """
    Create a combined relevance table out of a list of tuples consisting of a target label
    and its corresponding relevance table.
    The combined relevance table containing the p_values for all target labels.

    :param relevance_tables_with_label: A list of tuples: label, relevance table
    :type relevance_tables_with_label: List[Tuple[Any, pd.DataFrame]]
    :return: The combined relevance table
    :rtype: pandas.DataFrame
    """
    def _append_label_to_p_value_column(a):
        label, df = a
        return df.rename(columns={'p_value': 'p_value_{}'.format(label)})

    def _combine(a, b):
        a.relevant |= b.relevant
        return a.join(b.iloc[:,3])

    relevance_tables = map(_append_label_to_p_value_column, relevance_tables_with_label)
    relevance_table = reduce(_combine, relevance_tables)
    relevance_table['p_value'] = relevance_table.iloc[:, 3:].values.min(axis=1)
    return relevance_table
