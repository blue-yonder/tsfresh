# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from __future__ import absolute_import
import pandas as pd
from tsfresh.feature_extraction import extract_features, FeatureExtractionSettings
from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import restrict_input_to_index, impute


def extract_relevant_features(timeseries_container, y, X=None,
                              feature_extraction_settings=None,
                              feature_selection_settings=None,
                              column_id=None, column_sort=None, column_kind=None, column_value=None):
    """
    High level convenience function to extract time series features from `timeseries_container`. Then return feature
    matrix `X` possibly augmented with features relevant with respect to target vector `y`.

    For more details see the documentation of :func:`~tsfresh.feature_extraction.extraction.extract_features` and
    :func:`~tsfresh.feature_selection.selection.select_features`.

    Examples
    ========

    >>> from tsfresh.examples import load_robot_execution_failures
    >>> from tsfresh import extract_relevant_features
    >>> df, y = load_robot_execution_failures()
    >>> X = extract_relevant_features(df, y, column_id='id', column_sort='time')

    :param timeseries_container: See parameter `timeseries_container` in :func:`~tsfresh.feature_extraction.extraction.extract_features`
    :param y: See parameter `y` in :func:`~tsfresh.feature_selection.selection.select_features`
    :param X: See parameter `X` in :func:`~tsfresh.feature_selection.selection.select_features`
    :param column_id: See parameter `column_id` in :func:`~tsfresh.feature_extraction.extraction.extract_features`
    :param column_sort: See parameter `column_sort` in :func:`~tsfresh.feature_extraction.extraction.extract_features`
    :param column_kind: See parameter `column_kind` in :func:`~tsfresh.feature_extraction.extraction.extract_features`
    :param column_value: See parameter `column_value` in :func:`~tsfresh.feature_extraction.extraction.extract_features`
    :param feature_extraction_settings: See parameter `feature_extraction_settings` in :func:`~tsfresh.feature_extraction.extraction.extract_features`
    :param feature_selection_settings: See parameter `feature_selection_settings` in :func:`~tsfresh.feature_selection.selection.select_features`

    :return: Feature matrix X, possibly extended with relevant time series features.
    """
    if X is not None:
        timeseries_container = restrict_input_to_index(timeseries_container, column_id, X.index)

    if feature_extraction_settings is None:
        feature_extraction_settings = FeatureExtractionSettings()
        feature_extraction_settings.IMPUTE = impute

    X_ext = extract_features(timeseries_container, feature_extraction_settings=feature_extraction_settings,
                             column_id=column_id, column_sort=column_sort,
                             column_kind=column_kind, column_value=column_value)
    X_sel = select_features(X_ext, y, feature_selection_settings=feature_selection_settings)

    if X is None:
        X = X_sel
    else:
        X = pd.merge(X, X_sel, left_index=True, right_index=True, how="left")

    return X
