# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from __future__ import absolute_import
import pandas as pd
from tsfresh.feature_extraction import extract_features
from tsfresh import defaults
from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import restrict_input_to_index, impute


def extract_relevant_features(timeseries_container, y, X=None,
                              default_calculation_settings_mapping=None,
                              kind_to_calculation_settings_mapping=None,
                              column_id=None, column_sort=None, column_kind=None, column_value=None,
                              parallelization=defaults.PARALLELISATION, show_warnings=defaults.SHOW_WARNINGS,
                              disable_progressbar=defaults.DISABLE_PROGRESSBAR,
                              profile=defaults.PROFILING,
                              profiling_filename=defaults.PROFILING_FILENAME,
                              profiling_sorting=defaults.PROFILING_SORTING,
                              test_for_binary_target_binary_feature=defaults.TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
                              test_for_binary_target_real_feature=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE,
                              test_for_real_target_binary_feature=defaults.TEST_FOR_REAL_TARGET_BINARY_FEATURE,
                              test_for_real_target_real_feature=defaults.TEST_FOR_REAL_TARGET_REAL_FEATURE,
                              fdr_level=defaults.FDR_LEVEL, hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT,
                              write_selection_report=defaults.WRITE_SELECTION_REPORT, result_dir=defaults.RESULT_DIR,
                              n_processes=defaults.N_PROCESSES, chunksize=defaults.CHUNKSIZE):
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

    X_ext = extract_features(timeseries_container,
                             default_calculation_settings_mapping=default_calculation_settings_mapping,
                             kind_to_calculation_settings_mapping=kind_to_calculation_settings_mapping,
                             parallelization=parallelization, show_warnings=show_warnings,
                             disable_progressbar=disable_progressbar,
                             profile=profile,
                             profiling_filename=profiling_filename,
                             profiling_sorting=profiling_sorting,
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
