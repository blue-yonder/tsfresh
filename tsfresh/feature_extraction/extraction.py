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

import itertools
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

from tsfresh.feature_extraction import feature_calculators
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
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
    # TODO
    df_melt = pd.melt(timeseries_container.sort_values(column_sort).drop(column_sort, axis=1), id_vars=[column_id])

    # Use the standard setting if the user did not supply ones himself.
    if default_fc_parameters is None:
        default_fc_parameters = ComprehensiveFCParameters()

    _logger.info('Parallelizing feature calculation {}'.format(parallelization))

    # If requested, do profiling (advanced feature)
    if profile:
        profiler = profiling.start_profiling()

    with warnings.catch_warnings():
        if not show_warnings:
            warnings.simplefilter("ignore")
        else:
            warnings.simplefilter("default")

    result = _do_extraction(df_melt, column_id, default_fc_parameters, kind_to_fc_parameters)

    # Impute the result if requested
    if impute_function is not None:
        impute_function(result)

    # Turn off profiling if it was turned on
    if profile:
        profiling.end_profiling(profiler, filename=profiling_filename,
                                sorting=profiling_sorting)

    return result


def _do_extraction(df_melt, column_id, default_fc_parameters, kind_to_fc_parameters):
    data_in_chunks = [x + (y,) for x, y in df_melt.groupby([column_id, "variable"]).value]

    extraction_function = partial(_extract_named_function,
                                  default_fc_parameters=default_fc_parameters,
                                  kind_to_fc_parameters=kind_to_fc_parameters)

    # Map over all those chunks and extract the features on them
    result = map(extraction_function, data_in_chunks)

    # Flatten out the lists
    result = itertools.chain.from_iterable(result)

    # Return a dataframe in the typical form (id as index and feature names as columns)
    result = pd.DataFrame(list(result)).pivot("id", "variable", "value")

    return result


def _convert_to_output_format(param):
    return "_".join(str(key) + "_" + str(value) for key, value in param.items())


def _extract_named_function(chunk, default_fc_parameters, kind_to_fc_parameters):
    chunk_id, chunk_kind, data = chunk
    data = data.values

    if kind_to_fc_parameters and chunk_kind in kind_to_fc_parameters:
        fc_parameters = kind_to_fc_parameters[chunk_kind]
    else:
        fc_parameters = default_fc_parameters

    def _f():
        for function_name, parameter_list in fc_parameters.items():
            func = getattr(feature_calculators, function_name)

            if func.fctype == "combiner":
                result = func(data, param=parameter_list)
            else:
                if parameter_list:
                    result = ((_convert_to_output_format(param), func(data, **param)) for param in parameter_list)
                else:
                    result = [("", func(data))]

            for key, item in result:
                yield {"variable": chunk_kind + "__" + func.__name__ + "__" + str(key), "value": item, "id": chunk_id}

    return list(_f())


def _extract_features_per_kind(kind_to_df_map,
                               column_id, column_value,
                               default_fc_parameters,
                               kind_to_fc_parameters=None,
                               chunksize=defaults.CHUNKSIZE,
                               n_processes=defaults.N_PROCESSES, show_warnings=defaults.SHOW_WARNINGS,
                               disable_progressbar=defaults.DISABLE_PROGRESSBAR,
                               impute_function=defaults.IMPUTE_FUNCTION,
                               serial=False):
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

    :param serial: Run in serial instead of in parallel mode for performance testing
    :type serial: bool

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

    if serial:
        map_function = map
    else:
        map_function = partial(pool.imap_unordered, chunksize=chunksize)

    extracted_features = tqdm(map_function(partial_extract_features_for_one_time_series, kind_to_df_map.items()),
                              total=total_number_of_expected_results,
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
