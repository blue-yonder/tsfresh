# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import pandas as pd

from tsfresh import defaults
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import (
    get_ids,
    impute,
    restrict_input_to_index,
)


def extract_relevant_features(
    timeseries_container,
    y,
    X=None,
    default_fc_parameters=None,
    kind_to_fc_parameters=None,
    column_id=None,
    column_sort=None,
    column_kind=None,
    column_value=None,
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
    n_jobs=defaults.N_PROCESSES,
    distributor=None,
    chunksize=defaults.CHUNKSIZE,
    ml_task="auto",
):
    """
    High level convenience function to extract time series features from `timeseries_container`. Then return feature
    matrix `X` possibly augmented with relevant features with respect to target vector `y`.

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

    :param column_id: The name of the id column to group by. Please see :ref:`data-formats-label`.
    :type column_id: str

    :param column_sort: The name of the sort column. Please see :ref:`data-formats-label`.
    :type column_sort: str

    :param column_kind: The name of the column keeping record on the kind of the value.
            Please see :ref:`data-formats-label`.
    :type column_kind: str

    :param column_value: The name for the column keeping the value itself. Please see :ref:`data-formats-label`.
    :type column_value: str

    :param chunksize: The size of one chunk that is submitted to the worker
        process for the parallelisation.  Where one chunk is defined as a
        singular time series for one id and one kind. If you set the chunksize
        to 10, then it means that one task is to calculate all features for 10
        time series.  If it is set it to None, depending on distributor,
        heuristics are used to find the optimal chunksize. If you get out of
        memory exceptions, you can try it with the dask distributor and a
        smaller chunksize.
    :type chunksize: None or int

    :param n_jobs: The number of processes to use for parallelization. If zero, no parallelization is used.
    :type n_jobs: int

    :param distributor: Advanced parameter: set this to a class name that you want to use as a
             distributor. See the utilities/distribution.py for more information. Leave to None, if you want
             TSFresh to choose the best distributor.
    :type distributor: class

    :param show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
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

    :param ml_task: The intended machine learning task. Either `'classification'`, `'regression'` or `'auto'`.
                    Defaults to `'auto'`, meaning the intended task is inferred from `y`.
                    If `y` has a boolean, integer or object dtype, the task is assumed to be classification,
                    else regression.
    :type ml_task: str

    :return: Feature matrix X, possibly extended with relevant time series features.
    """

    assert isinstance(
        y, pd.Series
    ), "y needs to be a pandas.Series, received type: {}.".format(type(y))
    assert (
        len(set(y)) > 1
    ), "Feature selection is only possible if more than 1 label/class is provided"

    if X is not None:
        timeseries_container = restrict_input_to_index(
            timeseries_container, column_id, X.index
        )

    ids_container = get_ids(df_or_dict=timeseries_container, column_id=column_id)
    ids_y = set(y.index)
    if ids_container != ids_y:
        if len(ids_container - ids_y) > 0:
            raise ValueError(
                "The following ids are in the time series container but are missing in y: "
                "{}".format(ids_container - ids_y)
            )
        if len(ids_y - ids_container) > 0:
            raise ValueError(
                "The following ids are in y but are missing inside the time series container: "
                "{}".format(ids_y - ids_container)
            )

    X_ext = extract_features(
        timeseries_container,
        default_fc_parameters=default_fc_parameters,
        kind_to_fc_parameters=kind_to_fc_parameters,
        show_warnings=show_warnings,
        disable_progressbar=disable_progressbar,
        profile=profile,
        profiling_filename=profiling_filename,
        profiling_sorting=profiling_sorting,
        n_jobs=n_jobs,
        column_id=column_id,
        column_sort=column_sort,
        column_kind=column_kind,
        column_value=column_value,
        distributor=distributor,
        impute_function=impute,
    )

    X_sel = select_features(
        X_ext,
        y,
        test_for_binary_target_binary_feature=test_for_binary_target_binary_feature,
        test_for_binary_target_real_feature=test_for_binary_target_real_feature,
        test_for_real_target_binary_feature=test_for_real_target_binary_feature,
        test_for_real_target_real_feature=test_for_real_target_real_feature,
        fdr_level=fdr_level,
        hypotheses_independent=hypotheses_independent,
        n_jobs=n_jobs,
        show_warnings=show_warnings,
        chunksize=chunksize,
        ml_task=ml_task,
    )

    if X is None:
        X = X_sel
    else:
        X = pd.merge(X, X_sel, left_index=True, right_index=True, how="left")

    return X
