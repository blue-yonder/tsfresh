# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

# todo: complement this docstring and reference to christ paper
"""
Contains the methods from the following paper about the FRESH algorithm [2]_

Fresh is based on hypothesis tests that individually check the significance of every generated feature on the target.
It makes sure that only features are kept, that are relevant for the regression or classification task at hand. FRESH
decide between four settings depending if the features and target are binary or not.

The four functions are named

1. :func:`~tsfresh.feature_selection.significance_tests.target_binary_feature_binary_test`:
   Target and feature are both binary
2. :func:`~tsfresh.feature_selection.significance_tests.target_binary_feature_real_test`:
   Target is binary and feature real
3. :func:`~tsfresh.feature_selection.significance_tests.target_real_feature_binary_test`:
   Target is real and the feature is binary
4. :func:`~tsfresh.feature_selection.significance_tests.target_real_feature_real_test`:
   Target and feature are both real

References
----------

.. [2] Christ, M., Kempa-Liehr, A.W. and Feindt, M. (2016).
     Distributed and parallel time series feature extraction for industrial big data applications.
     ArXiv e-prints: 1610.07717
     https://arxiv.org/abs/1610.07717


"""

import warnings
from builtins import str

import numpy as np
import pandas as pd
from scipy import stats


def target_binary_feature_binary_test(x, y):
    """
    Calculate the feature significance of a binary feature to a binary target as a p-value.
    Use the two-sided univariate fisher test from :func:`~scipy.stats.fisher_exact` for this.

    :param x: the binary feature vector
    :type x: pandas.Series

    :param y: the binary target vector
    :type y: pandas.Series

    :return: the p-value of the feature significance test. Lower p-values indicate a higher feature significance
    :rtype: float

    :raise: ``ValueError`` if the target or the feature is not binary.
    """
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)

    # Check for correct value range
    __check_for_binary_feature(x)
    __check_for_binary_target(y)

    # Extract the unique values
    x0, x1 = np.unique(x.values)
    y0, y1 = np.unique(y.values)

    # Calculate contingency table
    n_y1_x0 = np.sum(y[x == x0] == y1)
    n_y0_x0 = len(y[x == x0]) - n_y1_x0
    n_y1_x1 = np.sum(y[x == x1] == y1)
    n_y0_x1 = len(y[x == x1]) - n_y1_x1

    table = np.array([[n_y1_x1, n_y1_x0], [n_y0_x1, n_y0_x0]])

    # Perform the Fisher test
    oddsratio, p_value = stats.fisher_exact(table, alternative="two-sided")

    return p_value


def target_binary_feature_real_test(x, y, test):
    """
    Calculate the feature significance of a real-valued feature to a binary target as a p-value.
    Use either the `Mann-Whitney U` or `Kolmogorov Smirnov` from  :func:`~scipy.stats.mannwhitneyu` or
    :func:`~scipy.stats.ks_2samp` for this.

    :param x: the real-valued feature vector
    :type x: pandas.Series

    :param y: the binary target vector
    :type y: pandas.Series

    :param test: The significance test to be used. Either ``'mann'`` for the Mann-Whitney-U test
                 or ``'smir'`` for the Kolmogorov-Smirnov test
    :type test: str

    :return: the p-value of the feature significance test. Lower p-values indicate a higher feature significance
    :rtype: float

    :raise: ``ValueError`` if the target is not binary.
    """
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)

    # Check for correct value range
    __check_for_binary_target(y)

    # Extract the unique values
    y0, y1 = np.unique(y.values)

    # Divide feature according to target
    x_y1 = x[y == y1]
    x_y0 = x[y == y0]

    if test == "mann":
        # Perform Mann-Whitney-U test
        U, p_mannwhitu = stats.mannwhitneyu(
            x_y1, x_y0, use_continuity=True, alternative="two-sided"
        )
        return p_mannwhitu
    elif test == "smir":
        # Perform Kolmogorov-Smirnov test
        KS, p_ks = stats.ks_2samp(x_y1, x_y0)
        return p_ks
    else:
        raise ValueError(
            "Please use a valid entry for test_for_binary_target_real_feature. "
            + "Valid entries are 'mann' and 'smir'."
        )


def target_real_feature_binary_test(x, y):
    """
    Calculate the feature significance of a binary feature to a real-valued target as a p-value.
    Use the `Kolmogorov-Smirnov` test from from :func:`~scipy.stats.ks_2samp` for this.

    :param x: the binary feature vector
    :type x: pandas.Series

    :param y: the real-valued target vector
    :type y: pandas.Series

    :return: the p-value of the feature significance test. Lower p-values indicate a higher feature significance.
    :rtype: float

    :raise: ``ValueError`` if the feature is not binary.
    """
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)

    # Check for correct value range
    __check_for_binary_feature(x)

    # Extract the unique values
    x0, x1 = np.unique(x.values)

    # Divide target according to feature
    y_x1 = y[x == x1]
    y_x0 = y[x == x0]

    # Perform Kolmogorov-Smirnov test
    KS, p_value = stats.ks_2samp(y_x1, y_x0)

    return p_value


def target_real_feature_real_test(x, y):
    """
    Calculate the feature significance of a real-valued feature to a real-valued target as a p-value.
    Use `Kendall's tau` from :func:`~scipy.stats.kendalltau` for this.

    :param x: the real-valued feature vector
    :type x: pandas.Series

    :param y: the real-valued target vector
    :type y: pandas.Series

    :return: the p-value of the feature significance test. Lower p-values indicate a higher feature significance.
    :rtype: float
    """
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)

    tau, p_value = stats.kendalltau(x, y, method="asymptotic")
    return p_value


def __check_if_pandas_series(x, y):
    """
    Helper function to check if both x and y are pandas.Series. If not, raises a ``TypeError``.

    :param x: the first object to check.
    :type x: Any

    :param y: the second object to check.
    :type y: Any

    :return: None
    :rtype: None

    :raise: ``TypeError`` if one of the objects is not a pandas.Series.
    """
    if not isinstance(x, pd.Series):
        raise TypeError("x should be a pandas Series")
    if not isinstance(y, pd.Series):
        raise TypeError("y should be a pandas Series")
    if not list(y.index) == list(x.index):
        raise ValueError("X and y need to have the same index!")


def __check_for_binary_target(y):
    """
    Helper function to check if a target column is binary.
    Checks if only the values true and false (or 0 and 1) are present in the values.

    :param y: the values to check for.
    :type y: pandas.Series or numpy.array

    :return: None
    :rtype: None

    :raises: ``ValueError`` if the values are not binary.
    """
    if not set(y) == {0, 1}:
        if len(set(y)) > 2:
            raise ValueError("Target is not binary!")

        warnings.warn(
            "The binary target should have "
            "values 1 and 0 (or True and False). "
            "Instead found" + str(set(y)),
            RuntimeWarning,
        )


def __check_for_binary_feature(x):
    """
    Helper function to check if a feature column is binary.
    Checks if only the values true and false (or 0 and 1) are present in the values.

    :param y: the values to check for.
    :type y: pandas.Series or numpy.array

    :return: None
    :rtype: None

    :raises: ``ValueError`` if the values are not binary.
    """
    if not set(x) == {0, 1}:
        if len(set(x)) > 2:
            raise ValueError(
                "[target_binary_feature_binary_test] Feature is not binary!"
            )

        warnings.warn(
            "A binary feature should have only "
            "values 1 and 0 (incl. True and False). "
            "Instead found " + str(set(x)) + " in feature ''" + str(x.name) + "''.",
            RuntimeWarning,
        )


def _check_for_nans(x, y):
    """
    Helper function to check if target or feature contains NaNs.
    :param x: A feature
    :type x: pandas.Series
    :param y: The target
    :type y: pandas.Series
    :raises: `ValueError` if target or feature contains NaNs.
    """
    if np.isnan(x.values).any():
        raise ValueError("Feature {} contains NaN values".format(x.name))
    elif np.isnan(y.values).any():
        raise ValueError("Target contains NaN values")
