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
from tsfresh.feature_selection.settings import FeatureSignificanceTestsSettings
from tsfresh.utilities.dataframe_functions import check_for_nans_in_columns
from tsfresh.feature_selection.feature_selector import check_fs_sig_bh


def select_features(X, y, feature_selection_settings=None):
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

    :param y: Target vector which is needed to test, which features are relevant. Can be binary or real-valued.
    :type y: pandas.Series or numpy.ndarray

    :param feature_selection_settings: The settings to control the feature selection algorithms. See
           :class:`~tsfresh.feature_selection.settings.py` for more information. If none is passed, the defaults
           will be used.
    :type feature_selection_settings: FeatureSignificanceTestsSettings

    :return: The same DataFrame as X, but possibly with reduced number of columns ( = features).
    :rtype: pandas.DataFrame

    :raises: ``ValueError`` when the target vector does not fit to the feature matrix.
    """
    check_for_nans_in_columns(X)

    if feature_selection_settings is None:
        feature_selection_settings = FeatureSignificanceTestsSettings()

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

    df_bh = check_fs_sig_bh(X, y, feature_selection_settings)

    return X.loc[:, df_bh[df_bh.rejected].Feature]
