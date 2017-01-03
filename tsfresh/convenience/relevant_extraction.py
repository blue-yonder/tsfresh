# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from __future__ import absolute_import
import pandas as pd
from tsfresh.feature_extraction import extract_features, FeatureExtractionSettings
from tsfresh.feature_selection import select_features, settings
from tsfresh.utilities.dataframe_functions import restrict_input_to_index, impute


def extract_relevant_features(timeseries_container, y, X=None,
                              feature_extraction_settings=None,
                              column_id=None, column_sort=None, column_kind=None, column_value=None,
                              test_for_binary_target_binary_feature=settings.TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
                              test_for_binary_target_real_feature=settings.TEST_FOR_BINARY_TARGET_REAL_FEATURE,
                              test_for_real_target_binary_feature=settings.TEST_FOR_REAL_TARGET_BINARY_FEATURE,
                              test_for_real_target_real_feature=settings.TEST_FOR_REAL_TARGET_REAL_FEATURE,
                              fdr_level=settings.FDR_LEVEL, hypotheses_independent=settings.HYPOTHESES_INDEPENDENT,
                              write_selection_report=settings.WRITE_SELECTION_REPORT, result_dir=settings.RESULT_DIR,
                              n_processes=settings.N_PROCESSES, chunksize=settings.CHUNKSIZE):
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

    X_ext = extract_features(timeseries_container, feature_extraction_settings=feature_extraction_settings,
                             column_id=column_id, column_sort=column_sort,
                             column_kind=column_kind, column_value=column_value,
                             impute_function=impute)
    X_sel = select_features(X_ext, y, test_for_binary_target_binary_feature=test_for_binary_target_binary_feature,
                            test_for_binary_target_real_feature=test_for_binary_target_real_feature,
                            test_for_real_target_binary_feature=test_for_real_target_binary_feature,
                            test_for_real_target_real_feature=test_for_real_target_real_feature,
                            fdr_level=fdr_level, hypotheses_independent=hypotheses_independent,
                            write_selection_report=write_selection_report, result_dir=result_dir,
                            n_processes=n_processes, chunksize=chunksize)

    if X is None:
        X = X_sel
    else:
        X = pd.merge(X, X_sel, left_index=True, right_index=True, how="left")

    return X
