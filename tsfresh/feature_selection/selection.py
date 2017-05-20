# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This module contains the filtering process for the extracted features. The filtering procedure can also be used on
other features that are not based on time series.
"""

from __future__ import absolute_import

import pandas as pd
import numpy as np
from tsfresh import defaults
from tsfresh.utilities.dataframe_functions import check_for_nans_in_columns
from tsfresh.feature_selection.feature_selector import check_fs_sig_bh


def select_features(X, y, test_for_binary_target_binary_feature=defaults.TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
                    test_for_binary_target_real_feature=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE,
                    test_for_real_target_binary_feature=defaults.TEST_FOR_REAL_TARGET_BINARY_FEATURE,
                    test_for_real_target_real_feature=defaults.TEST_FOR_REAL_TARGET_REAL_FEATURE,
                    fdr_level=defaults.FDR_LEVEL, hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT,
                    n_processes=defaults.N_PROCESSES, chunksize=defaults.CHUNKSIZE):
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

    :param n_processes: Number of processes to use during the p-value calculation
    :type n_processes: int

    :param chunksize: Size of the chunks submitted to the worker processes
    :type chunksize: int

    :return: The same DataFrame as X, but possibly with reduced number of columns ( = features).
    :rtype: pandas.DataFrame

    :raises: ``ValueError`` when the target vector does not fit to the feature matrix.
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

    df_bh = check_fs_sig_bh(X, y, n_processes, chunksize, fdr_level, hypotheses_independent,
                            test_for_binary_target_real_feature)

    return X.loc[:, df_bh[df_bh.rejected].Feature]
