# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This module contains the main function to interact with tsfresh: extract features
"""

from __future__ import absolute_import, division

import itertools
import logging
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from tsfresh import defaults
from tsfresh.feature_extraction import feature_calculators
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.utilities import dataframe_functions, profiling
from tsfresh.utilities.string_manipulation import convert_to_output_format

_logger = logging.getLogger(__name__)


def extract_features(timeseries_container, default_fc_parameters=None,
                     kind_to_fc_parameters=None,
                     column_id=None, column_sort=None, column_kind=None, column_value=None,
                     chunksize=defaults.CHUNKSIZE,
                     n_jobs=defaults.N_PROCESSES, show_warnings=defaults.SHOW_WARNINGS,
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

    :param n_jobs: The number of processes to use for parallelization. If zero, no parallelization is used.
    :type n_jobs: int

    :param chunksize: The size of one chunk for the parallelisation
    :type chunksize: None or int

    :param: show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
    :type show_warnings: bool

    :param disable_progressbar: Do not show a progressbar while doing the calculation.
    :type disable_progressbar: bool

    :param impute_function: None, if no imputing should happen or the function to call for imputing.
    :type impute_function: None or callable

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
    df_melt, column_id, column_kind, column_value = \
        dataframe_functions._normalize_input_to_internal_representation(timeseries_container=timeseries_container,
                                                                        column_id=column_id, column_kind=column_kind,
                                                                        column_sort=column_sort,
                                                                        column_value=column_value)

    # Use the standard setting if the user did not supply ones himself.
    if default_fc_parameters is None:
        default_fc_parameters = ComprehensiveFCParameters()

    # If requested, do profiling (advanced feature)
    if profile:
        profiler = profiling.start_profiling()

    with warnings.catch_warnings():
        if not show_warnings:
            warnings.simplefilter("ignore")
        else:
            warnings.simplefilter("default")

        result = _do_extraction(df=df_melt,
                                column_id=column_id, column_value=column_value, column_kind=column_kind,
                                n_jobs=n_jobs, chunksize=chunksize,
                                disable_progressbar=disable_progressbar,
                                default_fc_parameters=default_fc_parameters, kind_to_fc_parameters=kind_to_fc_parameters)

        # Impute the result if requested
        if impute_function is not None:
            impute_function(result)

    # Turn off profiling if it was turned on
    if profile:
        profiling.end_profiling(profiler, filename=profiling_filename,
                                sorting=profiling_sorting)

    return result


def _do_extraction(df, column_id, column_value, column_kind,
                   default_fc_parameters, kind_to_fc_parameters,
                   n_jobs, chunksize, disable_progressbar):
    """
    Wrapper around the _do_extraction_on_chunk, which calls it on all chunks in the data frame.
    A chunk is a subset of the data, with a given kind and id - so a single time series.

    The data is separated out into those single time series and the _do_extraction_on_chunk is
    called on each of them. The results are then combined into a single pandas DataFrame.

    The call is either happening in parallel or not and is showing a progress bar or not depending
    on the given flags.

    :param df: The dataframe in the normalized format which is used for extraction.
    :type df: pd.DataFrame

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

    :param column_kind: The name of the column keeping record on the kind of the value.
    :type column_kind: str

    :param column_value: The name for the column keeping the value itself.
    :type column_value: str

    :param chunksize: The size of one chunk for the parallelization
    :type chunksize: None or int

    :param n_jobs: The number of processes to use for parallelization. If zero, no parallelization is used.
    :type n_jobs: int

    :param disable_progressbar: Do not show a progressbar while doing the calculation.
    :type disable_progressbar: bool

    :return: the extracted features
    :rtype: pd.DataFrame
    """
    data_in_chunks = [x + (y,) for x, y in df.groupby([column_id, column_kind])[column_value]]

    total_number_of_expected_results = len(data_in_chunks)

    if n_jobs == 0:
        map_function = map
    else:
        pool = Pool(n_jobs)

        if not chunksize:
            chunksize = _calculate_best_chunksize(data_in_chunks, n_jobs)

        map_function = partial(pool.imap_unordered, chunksize=chunksize)

    extraction_function = partial(_do_extraction_on_chunk,
                                  default_fc_parameters=default_fc_parameters,
                                  kind_to_fc_parameters=kind_to_fc_parameters)

    # Map over all those chunks and extract the features on them
    result = tqdm(map_function(extraction_function, data_in_chunks),
                  total=total_number_of_expected_results,
                  desc="Feature Extraction", disable=disable_progressbar)

    # Flatten out the lists
    result = itertools.chain.from_iterable(result)

    # Return a dataframe in the typical form (id as index and feature names as columns)
    result = pd.DataFrame(list(result), dtype=np.float)

    if len(result) != 0:
        result = result.pivot("id", "variable", "value")
        result.index = result.index.astype(df[column_id].dtype)

    return result


def _do_extraction_on_chunk(chunk, default_fc_parameters, kind_to_fc_parameters):
    """
    Main function of this module: use the feature calculators defined in the
    default_fc_parameters or kind_to_fc_parameters parameters and extract all
    features on the chunk.

    The chunk consists of the chunk id, the chunk kind and the data (as a Series),
    which is then converted to a numpy array - so a single time series.

    Returned is a list of the extracted features. Each one is a dictionary consisting of
    { "variable": the feature name in the format <kind>__<feature>__<parameters>,
      "value": the number value of the feature,
      "id": the id of the chunk }

    The <parameters> are in the form described in :mod:`~tsfresh.utilities.string_manipulation`.

    :param chunk: A tuple of sample_id, kind, data
    :param default_fc_parameters: A dictionary of feature calculators.
    :param kind_to_fc_parameters: A dictionary of fc_parameters for special kinds or None.
    :return: A list of calculated features.
    """
    sample_id, kind, data = chunk
    data = data.values

    if kind_to_fc_parameters and kind in kind_to_fc_parameters:
        fc_parameters = kind_to_fc_parameters[kind]
    else:
        fc_parameters = default_fc_parameters

    def _f():
        for function_name, parameter_list in fc_parameters.items():
            func = getattr(feature_calculators, function_name)

            if func.fctype == "combiner":
                result = func(data, param=parameter_list)
            else:
                if parameter_list:
                    result = ((convert_to_output_format(param), func(data, **param)) for param in parameter_list)
                else:
                    result = [("", func(data))]

            for key, item in result:
                feature_name = str(kind) + "__" + func.__name__
                if key:
                    feature_name += "__" + str(key)
                yield {"variable": feature_name, "value": item, "id": sample_id}

    return list(_f())


def _calculate_best_chunksize(iterable_list, n_jobs):
    """
    Helper function to calculate the best chunksize for a given number of elements to calculate.

    The formula is more or less an empirical result.
    :param iterable_list: A list which defines how many calculations there need to be.
    :param n_jobs: The number of processes that will be used in the calculation.
    :return: The chunksize which should be used.

    TODO: Investigate which is the best chunk size for different settings.
    """
    chunksize, extra = divmod(len(iterable_list), n_jobs * 5)
    if extra:
        chunksize += 1
    return chunksize
