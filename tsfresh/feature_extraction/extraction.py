# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This module contains the main function to interact with tsfresh: extract features
"""

from __future__ import absolute_import, division

import logging
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from builtins import str
from multiprocessing import Pool
from functools import partial
from six.moves.queue import Queue
import logging
import pandas as pd
import numpy as np

from tqdm import tqdm
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, get_aggregate_functions, get_apply_functions
from tsfresh import defaults
from tsfresh.utilities import dataframe_functions, profiling

_logger = logging.getLogger(__name__)


def extract_features(timeseries_container, default_fc_parameters=None,
                     kind_to_fc_parameters=None,
                     column_id=None, column_sort=None, column_kind=None, column_value=None,
                     parallelization=None, chunksize=defaults.CHUNKSIZE,
                     n_processes=defaults.N_PROCESSES, show_warnings=defaults.SHOW_WARNINGS,
                     disable_progressbar=defaults.DISABLE_PROGRESSBAR,
                     impute_function=defaults.IMPUTE_FUNCTION,
                     profile=defaults.PROFILING,
                     profiling_filename=defaults.PROFILING_FILENAME,
                     profiling_sorting=defaults.PROFILING_SORTING):
    """
    Extract features from

    * a :class:`pandas.DataFrame` containing the different time series

    or

    * a dictionary of :class:`pandas.DataFrame` each containing one type of time series

    In both cases a :class:`pandas.DataFrame` with the calculated features will be returned.

    For a list of all the calculated time series features, please see the
    :class:`~tsfresh.feature_extraction.settings.ComprehensiveFCParameters` class,
    which is used to control which features with which parameters are calculated.

    For a detailed explanation of the different parameters and data formats please see :ref:`data-formats-label`.

    Examples
    ========

    >>> from tsfresh.examples import load_robot_execution_failures
    >>> from tsfresh import extract_features
    >>> df, _ = load_robot_execution_failures()
    >>> X = extract_features(df, column_id='id', column_sort='time')

    :param timeseries_container: The pandas.DataFrame with the time series to compute the features for, or a
            dictionary of pandas.DataFrames.
    :type timeseries_container: pandas.DataFrame or dict

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
                            some general assumptions.
    :type parallelization: str

    :param chunksize: The size of one chunk for the parallelisation
    :type chunksize: None or int

    :param n_processes: The number of processes to use for parallelisation.
    :type n_processes: int

    :param: show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
    :type show_warnings: bool

    :param disable_progressbar: Do not show a progressbar while doing the calculation.
    :type disable_progressbar: bool

    :param impute_function: None, if no imputing should happen or the function to call for imputing.
    :type impute_function: None or function

    :param profile: Turn on profiling during feature extraction
    :type profile: bool

    :param profiling_sorting: How to sort the profiling results (see the documentation of the profiling package for
           more information)
    :type profiling_sorting: basestring

    :param profiling_filename: Where to save the profiling results.
    :type profiling_filename: basestring

    :return: The (maybe imputed) DataFrame containing extracted features.
    :rtype: pandas.DataFrame
    """
    import logging
    logging.basicConfig()
    
    # Always use the standardized way of storing the data.
    # See the function normalize_input_to_internal_representation for more information.
    kind_to_df_map, column_id, column_value = \
        dataframe_functions.normalize_input_to_internal_representation(df_or_dict=timeseries_container,
                                                                       column_id=column_id,
                                                                       column_sort=column_sort,
                                                                       column_kind=column_kind,
                                                                       column_value=column_value)

    # Use the standard setting if the user did not supply ones himself.
    if default_fc_parameters is None:
        default_fc_parameters = ComprehensiveFCParameters()

    # Choose the parallelization according to a rule-of-thumb
    if parallelization is None:
        parallelization = 'per_sample' if n_processes / 2 > len(kind_to_df_map) else 'per_kind'

    _logger.info('Parallelizing feature calculation {}'.format(parallelization))

    # If requested, do profiling (advanced feature)
    if profile:
        profiler = profiling.start_profiling()

    # Calculate the result
    if parallelization == 'per_kind':
        calculation_function = _extract_features_parallel_per_kind
    elif parallelization == 'per_sample':
        calculation_function = _extract_features_parallel_per_sample
    else:
        raise ValueError("Argument parallelization must be one of: 'per_kind', 'per_sample'")

    result = calculation_function(kind_to_df_map,
                                  default_fc_parameters=default_fc_parameters,
                                  kind_to_fc_parameters=kind_to_fc_parameters,
                                  column_id=column_id,
                                  column_value=column_value,
                                  chunksize=chunksize,
                                  n_processes=n_processes,
                                  show_warnings=show_warnings,
                                  disable_progressbar=disable_progressbar,
                                  impute_function=impute_function
                                  )

    # Turn off profiling if it was turned on
    if profile:
        profiling.end_profiling(profiler, filename=profiling_filename,
                                sorting=profiling_sorting)

    return result


def _extract_features_parallel_per_kind(kind_to_df_map,
                                        column_id, column_value,
                                        default_fc_parameters,
                                        kind_to_fc_parameters=None,
                                        chunksize=defaults.CHUNKSIZE,
                                        n_processes=defaults.N_PROCESSES, show_warnings=defaults.SHOW_WARNINGS,
                                        disable_progressbar=defaults.DISABLE_PROGRESSBAR,
                                        impute_function=defaults.IMPUTE_FUNCTION):
    """
    Parallelize the feature extraction per kind.

    :param kind_to_df_map: The time series to compute the features for in our internal format
    :type kind_to_df_map: dict of pandas.DataFrame

    :param column_id: The name of the id column to group by.
    :type column_id: str

    :param column_value: The name for the column keeping the value itself.
    :type column_value: str

    :param default_fc_parameters: mapping from feature calculator names to parameters. Only those names
           which are keys in this dict will be calculated. See the class:`ComprehensiveFCParameters` for
           more information.
    :type default_fc_parameters: dict

    :param kind_to_fc_parameters: mapping from kind names to objects of the same type as the ones for
            default_fc_parameters. If you put a kind as a key here, the fc_parameters
            object (which is the value), will be used instead of the default_fc_parameters.
    :type kind_to_fc_parameters: dict

    :param chunksize: The size of one chunk for the parallelisation
    :type chunksize: None or int

    :param n_processes: The number of processes to use for parallelisation.
    :type n_processes: int

    :param: show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
    :type show_warnings: bool

    :param disable_progressbar: Do not show a progressbar while doing the calculation.
    :type disable_progressbar: bool

    :param impute_function: None, if no imputing should happen or the function to call for imputing.
    :type impute_function: None or function

    :return: The (maybe imputed) DataFrame containing extracted features.
    :rtype: pandas.DataFrame
    """
    partial_extract_features_for_one_time_series = partial(_extract_features_for_one_time_series,
                                                           column_id=column_id,
                                                           column_value=column_value,
                                                           default_fc_parameters=default_fc_parameters,
                                                           kind_to_fc_parameters=kind_to_fc_parameters,
                                                           show_warnings=show_warnings)
    pool = Pool(n_processes)

    if not chunksize:
        chunksize = _calculate_best_chunksize(kind_to_df_map, n_processes)

    total_number_of_expected_results = len(kind_to_df_map)
    extracted_features = tqdm(pool.imap_unordered(partial_extract_features_for_one_time_series, kind_to_df_map.items(),
                                                  chunksize=chunksize), total=total_number_of_expected_results,
                              desc="Feature Extraction", disable=disable_progressbar)
    pool.close()

    # Concatenate all partial results
    result = pd.concat(extracted_features, axis=1, join='outer').astype(np.float64)

    # Impute the result if requested
    if impute_function is not None:
        impute_function(result)

    pool.join()
    return result


def _extract_features_parallel_per_sample(kind_to_df_map,
                                          column_id, column_value,
                                          default_fc_parameters,
                                          kind_to_fc_parameters=None,
                                          chunksize=defaults.CHUNKSIZE,
                                          n_processes=defaults.N_PROCESSES, show_warnings=defaults.SHOW_WARNINGS,
                                          disable_progressbar=defaults.DISABLE_PROGRESSBAR,
                                          impute_function=defaults.IMPUTE_FUNCTION):
    """
    Parallelize the feature extraction per kind and per sample.

    As the splitting of the dataframes per kind along column_id is quite costly, we settled for an async map in this
    function. The result objects are temporarily stored in a fifo queue from which they can be retrieved in order
    of submission.

    :param kind_to_df_map: The time series to compute the features for in our internal format
    :type kind_to_df_map: dict of pandas.DataFrame

    :param column_id: The name of the id column to group by.
    :type column_id: str

    :param column_value: The name for the column keeping the value itself.
    :type column_value: str

    :param default_fc_parameters: mapping from feature calculator names to parameters. Only those names
           which are keys in this dict will be calculated. See the class:`ComprehensiveFCParameters` for
           more information.
    :type default_fc_parameters: dict

    :param kind_to_fc_parameters: mapping from kind names to objects of the same type as the ones for
            default_fc_parameters. If you put a kind as a key here, the fc_parameters
            object (which is the value), will be used instead of the default_fc_parameters.
    :type kind_to_fc_parameters: dict

    :param chunksize: The size of one chunk for the parallelisation
    :type chunksize: None or int

    :param n_processes: The number of processes to use for parallelisation.
    :type n_processes: int

    :param: show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
    :type show_warnings: bool

    :param disable_progressbar: Do not show a progressbar while doing the calculation.
    :type disable_progressbar: bool

    :param impute_function: None, if no imputing should happen or the function to call for imputing.
    :type impute_function: None or function

    :return: The (maybe imputed) DataFrame containing extracted features.
    :rtype: pandas.DataFrame
    """
    partial_extract_features_for_one_time_series = partial(_extract_features_for_one_time_series,
                                                           column_id=column_id,
                                                           column_value=column_value,
                                                           default_fc_parameters=default_fc_parameters,
                                                           kind_to_fc_parameters=kind_to_fc_parameters,
                                                           show_warnings=show_warnings)
    pool = Pool(n_processes)
    total_number_of_expected_results = 0

    # Submit map jobs per kind per sample
    results_fifo = Queue()

    for kind, df_kind in kind_to_df_map.items():
        df_grouped_by_id = df_kind.groupby(column_id)

        total_number_of_expected_results += len(df_grouped_by_id)

        if not chunksize:
            chunksize = _calculate_best_chunksize(df_grouped_by_id, n_processes)

        results_fifo.put(
            pool.imap_unordered(
                partial_extract_features_for_one_time_series,
                [(kind, df_group) for _, df_group in df_grouped_by_id],
                chunksize=chunksize
            )
        )

    pool.close()

    # Wait for the jobs to complete and concatenate the partial results
    dfs_per_kind = []

    # Do this all with a progress bar
    with tqdm(total=total_number_of_expected_results, desc="Feature Extraction",
              disable=disable_progressbar) as progress_bar:
        # We need some sort of measure, when a new result is there. So we wrap the
        # map_results into another iterable which updates the progress bar each time,
        # a new result is there
        def iterable_with_tqdm_update(queue, progress_bar):
            for element in queue:
                progress_bar.update(1)
                yield element

        result = pd.DataFrame()
        while not results_fifo.empty():
            map_result = results_fifo.get()
            dfs_kind = iterable_with_tqdm_update(map_result, progress_bar)
            df_tmp = pd.concat(dfs_kind, axis=0).astype(np.float64)

            # Impute the result if requested
            if impute_function is not None:
                impute_function(df_tmp)

            result = pd.concat([result, df_tmp], axis=1).astype(np.float64)

    pool.join()
    return result


def _calculate_best_chunksize(iterable_list, n_processes):
    """
    Helper function to calculate the best chunksize for a given number of elements to calculate.

    The formula is more or less an empirical result.
    :param iterable_list: A list which defines how many calculations there need to be.
    :param n_processes: The number of processes that will be used in the calculation.
    :return: The chunksize which should be used.

    TODO: Investigate which is the best chunk size for different settings.
    """
    chunksize, extra = divmod(len(iterable_list), n_processes * 5)
    if extra:
        chunksize += 1
    return chunksize


def _extract_features_for_one_time_series(prefix_and_dataframe, column_id, column_value,
                                          default_fc_parameters,
                                          kind_to_fc_parameters=None,
                                          show_warnings=defaults.SHOW_WARNINGS):
    """
    Extract time series features for a given data frame based on the passed settings.

    This is an internal function, please use the extract_features function.

    The `dataframe` is expected to have at least two columns: column_id and column_value. The data is grouped together
    by their column_id value and the time series features are calculated independently for each of the groups.
    As a result, the function returns a :class:`pandas.DataFrame` with the ids as an index and as many columns as there
    were features calculated.

    To distinguish the features from others, all the calculated columns are given the prefix passed in by column_prefix.

    For example, if you pass in a `dataframe` of shape

        +====+=======+=====+
        | id | value | ... |
        +====+=======+=====+
        | A  | 1     | ... |
        +----+-------+-----+
        | A  | 2     | ... |
        +----+-------+-----+
        | A  | 3     | ... |
        +----+-------+-----+
        | B  | 1     | ... |
        +----+-------+-----+
        | B  | 2     | ... |
        +----+-------+-----+
        | B  | 3     | ... |
        +----+-------+-----+

    with `column_id="id"`, `column_value="value"` and `column_prefix="prefix"` the resulting :class:`pandas.DataFrame`
    will have shape

        +=======+==================+==================+=====+==================+
        | Index | prefix_feature_1 | prefix_feature_2 | ... | prefix_feature_N |
        +=======+==================+==================+=====+==================+
        | A     | ...              | ...              | ... | ...              |
        +-------+------------------+------------------+-----+------------------+
        | B     | ...              | ...              | ... | ...              |
        +-------+------------------+------------------+-----+------------------+

    where N is the number of features that were calculated. Which features are calculated is controlled by the
    passed settings instance (see :class:`~tsfresh.feature_extraction.settings.ComprehensiveFCParameters` for a list of
    all possible features to calculate).

    The parameter `dataframe` is not allowed to have any NaN value in it. It is possible to have different numbers
    of values for different ids.

    :param prefix_and_dataframe: Tuple of column_prefix and dataframe
        column_prefix is the string that each extracted feature will be prefixed with (for better separation)
        dataframe with at least the columns column_id and column_value to extract the time
        series features for.
    :type prefix_and_dataframe: (str, DataFrame)

    :param column_id: The name of the column with the ids.
    :type column_id: str

    :param column_value: The name of the column with the values.
    :type column_value: str

    :param default_fc_parameters: mapping from feature calculator names to parameters. Only those names
           which are keys in this dict will be calculated. See the class:`ComprehensiveFCParameters` for
           more information.
    :type default_fc_parameters: dict

    :param kind_to_fc_parameters: mapping from kind names to objects of the same type as the ones for
            default_fc_parameters. If you put a kind as a key here, the fc_parameters
            object (which is the value), will be used instead of the default_fc_parameters.
    :type kind_to_fc_parameters: dict

    :param show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
    :type show_warnings: bool

    :return: A dataframe with the extracted features as the columns (prefixed with column_prefix) and as many
        rows as their are unique values in the id column.
    """
    if kind_to_fc_parameters is None:
        kind_to_fc_parameters = {}

    column_prefix, dataframe = prefix_and_dataframe
    column_prefix = str(column_prefix)

    # Ensure features are calculated on float64
    dataframe[column_value] = dataframe[column_value].astype(np.float64)

    # If there are no special settings for this column_prefix, use the default ones.
    if column_prefix in kind_to_fc_parameters:
        fc_parameters = kind_to_fc_parameters[column_prefix]
    else:
        fc_parameters = default_fc_parameters

    with warnings.catch_warnings():
        if not show_warnings:
            warnings.simplefilter("ignore")
        else:
            warnings.simplefilter("default")

        # Calculate the aggregation functions
        column_name_to_aggregate_function = get_aggregate_functions(fc_parameters, column_prefix)

        if column_name_to_aggregate_function:
            extracted_features = dataframe.groupby(column_id)[column_value].aggregate(column_name_to_aggregate_function)
        else:
            extracted_features = pd.DataFrame(index=dataframe[column_id].unique())

        # Calculate the apply functions
        apply_functions = get_apply_functions(fc_parameters, column_prefix)

        if apply_functions:
            list_of_extracted_feature_dataframes = [extracted_features]
            for apply_function, kwargs in apply_functions:
                current_result = dataframe.groupby(column_id)[column_value].apply(apply_function, **kwargs).unstack()
                if len(current_result) > 0:
                    list_of_extracted_feature_dataframes.append(current_result)

            if len(list_of_extracted_feature_dataframes) > 0:
                extracted_features = pd.concat(list_of_extracted_feature_dataframes, axis=1,
                                               join_axes=[extracted_features.index])

        return extracted_features
