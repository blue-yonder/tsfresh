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
                              default_fc_parameters=None,
                              kind_to_fc_parameters=None,
                              column_id=None, column_sort=None, column_kind=None, column_value=None,
                              parallelization=defaults.PARALLELISATION,
                              show_warnings=defaults.SHOW_WARNINGS,
                              disable_progressbar=defaults.DISABLE_PROGRESSBAR,
                              profile=defaults.PROFILING,
                              profiling_filename=defaults.PROFILING_FILENAME,
                              profiling_sorting=defaults.PROFILING_SORTING,
                              test_for_binary_target_binary_feature=defaults.TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
                              test_for_binary_target_real_feature=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE,
                              test_for_real_target_binary_feature=defaults.TEST_FOR_REAL_TARGET_BINARY_FEATURE,
                              test_for_real_target_real_feature=defaults.TEST_FOR_REAL_TARGET_REAL_FEATURE,
                              fdr_level=defaults.FDR_LEVEL,
                              hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT,
                              n_processes=defaults.N_PROCESSES,
                              chunksize=defaults.CHUNKSIZE):
    """
    High level convenience function to extract time series features from `timeseries_container`. Then return feature
    matrix `X` possibly augmented with relevent features with respect to target vector `y`.

    For more details see the documentation of :func:`~tsfresh.feature_extraction.extraction.extract_features` and
    :func:`~tsfresh.feature_selection.selection.select_features`.

    Examples
    ========

    >>> from tsfresh.examples import load_robot_execution_failures
    >>> from tsfresh import extract_relevant_features
    >>> df, y = load_robot_execution_failures()
    >>> X = extract_relevant_features(df, y, column_id='id', column_sort='time')

    :param timeseries_container: The pandas.DataFrame with the time series to compute the features for, or a
            dictionary of pandas.DataFrames.
            See :func:`~tsfresh.feature_extraction.extraction.extract_features`.

    :param X: A DataFrame containing additional features
    :type X: pandas.DataFrame

    :param y: The target vector
    :type y: pandas.Series

    :param default_fc_parameters: mapping from feature calculator names to parameters. Only those names
           which are keys in this dict will be calculated. See the class:`ComprehensiveFCParameters` for
           more information.
    :type default_fc_parameters: dict

    :param kind_to_fc_parameters: mapping from kind names to objects of the same type as the ones for
            default_fc_parameters. If you put a kind as a key here, the fc_parameters
            object (which is the value), will be used instead of the default_fc_parameters.
    :type kind_to_fc_parameters: dict

    :param column_id: The name of the id column to group by.
    :type column_id: str

    :param column_sort: The name of the sort column.
    :type column_sort: str

    :param column_kind: The name of the column keeping record on the kind of the value.
    :type column_kind: str

    :param column_value: The name for the column keeping the value itself.
    :type column_value: str

    :param parallelization: Either ``'per_sample'`` or ``'per_kind'``   , see
                            :func:`~tsfresh.feature_extraction.extraction._extract_features_parallel_per_sample`,
                            :func:`~tsfresh.feature_extraction.extraction._extract_features_parallel_per_kind` and
                            :ref:`parallelization-label` for details.
                            Choosing None makes the algorithm look for the best parallelization technique by applying
                            some general remarks.
    :type parallelization: str

    :param chunksize: The size of one chunk for the parallelisation
    :type chunksize: None or int

    :param n_processes: The number of processes to use for parallelisation.
    :type n_processes: int

    :param: show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
    :type show_warnings: bool

    :param disable_progressbar: Do not show a progressbar while doing the calculation.
    :type disable_progressbar: bool

    :param profile: Turn on profiling during feature extraction
    :type profile: bool

    :param profiling_sorting: How to sort the profiling results (see the documentation of the profiling package for
           more information)
    :type profiling_sorting: basestring

    :param profiling_filename: Where to save the profiling results.
    :type profiling_filename: basestring

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

    :param write_selection_report: Whether to store the selection report after the Benjamini Hochberg procedure has
                                   finished.
    :type write_selection_report: bool

    :param result_dir: Where to store the selection report
    :type result_dir: str

    :return: Feature matrix X, possibly extended with relevant time series features.
    """
    if X is not None:
        timeseries_container = restrict_input_to_index(timeseries_container, column_id, X.index)

    X_ext = extract_features(timeseries_container,
                             default_fc_parameters=default_fc_parameters,
                             kind_to_fc_parameters=kind_to_fc_parameters,
                             parallelization=parallelization,
                             show_warnings=show_warnings,
                             disable_progressbar=disable_progressbar,
                             profile=profile,
                             profiling_filename=profiling_filename,
                             profiling_sorting=profiling_sorting,
                             column_id=column_id, column_sort=column_sort,
                             column_kind=column_kind, column_value=column_value,
                             impute_function=impute)

    X_sel = select_features(X_ext, y,
                            test_for_binary_target_binary_feature=test_for_binary_target_binary_feature,
                            test_for_binary_target_real_feature=test_for_binary_target_real_feature,
                            test_for_real_target_binary_feature=test_for_real_target_binary_feature,
                            test_for_real_target_real_feature=test_for_real_target_real_feature,
                            fdr_level=fdr_level, hypotheses_independent=hypotheses_independent,
                            n_processes=n_processes,
                            chunksize=chunksize)

    if X is None:
        X = X_sel
    else:
        X = pd.merge(X, X_sel, left_index=True, right_index=True, how="left")

    return X
