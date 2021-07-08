# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This module contains the filtering process for the extracted features. The filtering procedure can also be used on
other features that are not based on time series.
"""

import numpy as np
import pandas as pd

from tsfresh import defaults
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import check_for_nans_in_columns


def select_features(
    X,
    y,
    test_for_binary_target_binary_feature=defaults.TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
    test_for_binary_target_real_feature=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE,
    test_for_real_target_binary_feature=defaults.TEST_FOR_REAL_TARGET_BINARY_FEATURE,
    test_for_real_target_real_feature=defaults.TEST_FOR_REAL_TARGET_REAL_FEATURE,
    fdr_level=defaults.FDR_LEVEL,
    hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT,
    n_jobs=defaults.N_PROCESSES,
    show_warnings=defaults.SHOW_WARNINGS,
    chunksize=defaults.CHUNKSIZE,
    ml_task="auto",
    multiclass=False,
    n_significant=1,
):
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

    :param test_for_binary_target_binary_feature: Which test to be used for binary target, binary feature
                                                  (currently unused)
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

    :param show_warnings: Show warnings during the p-value calculation (needed for debugging of calculators).
    :type show_warnings: bool

    :param chunksize: The size of one chunk that is submitted to the worker
        process for the parallelisation.  Where one chunk is defined as
        the data for one feature. If you set the chunksize
        to 10, then it means that one task is to filter 10 features.
        If it is set it to None, depending on distributor,
        heuristics are used to find the optimal chunksize. If you get out of
        memory exceptions, you can try it with the dask distributor and a
        smaller chunksize.
    :type chunksize: None or int

    :param ml_task: The intended machine learning task. Either `'classification'`, `'regression'` or `'auto'`.
                    Defaults to `'auto'`, meaning the intended task is inferred from `y`.
                    If `y` has a boolean, integer or object dtype, the task is assumed to be classification,
                    else regression.
    :type ml_task: str

    :param multiclass: Whether the problem is multiclass classification. This modifies the way in which features
                       are selected. Multiclass requires the features to be statistically significant for
                       predicting n_significant features.
    :type multiclass: bool

    :param n_significant: The number of classes for which features should be statistically significant predictors
                          to be regarded as 'relevant'. Only specify when multiclass=True
    :type n_significant: int

    :return: The same DataFrame as X, but possibly with reduced number of columns ( = features).
    :rtype: pandas.DataFrame

    :raises: ``ValueError`` when the target vector does not fit to the feature matrix
             or `ml_task` is not one of `'auto'`, `'classification'` or `'regression'`.
    """
    assert isinstance(X, pd.DataFrame), "Please pass features in X as pandas.DataFrame."
    check_for_nans_in_columns(X)
    assert isinstance(y, (pd.Series, np.ndarray)), (
        "The type of target vector y must be one of: " "pandas.Series, numpy.ndarray"
    )
    assert len(y) > 1, "y must contain at least two samples."
    assert len(X) == len(y), "X and y must contain the same number of samples."
    assert (
        len(set(y)) > 1
    ), "Feature selection is only possible if more than 1 label/class is provided"

    if isinstance(y, pd.Series) and set(X.index) != set(y.index):
        raise ValueError("Index of X and y must be identical if provided")

    if isinstance(y, np.ndarray):
        y = pd.Series(y, index=X.index)

    relevance_table = calculate_relevance_table(
        X,
        y,
        ml_task=ml_task,
        multiclass=multiclass,
        n_significant=n_significant,
        n_jobs=n_jobs,
        show_warnings=show_warnings,
        chunksize=chunksize,
        test_for_binary_target_real_feature=test_for_binary_target_real_feature,
        fdr_level=fdr_level,
        hypotheses_independent=hypotheses_independent,
    )

    relevant_features = relevance_table[relevance_table.relevant].feature

    return X.loc[:, relevant_features]
