# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This module contains the main function to interact with tsfresh: extract features
"""

from __future__ import absolute_import, division

import warnings
from builtins import str
from multiprocessing import Pool
from functools import partial
from six.moves.queue import Queue
import logging
import pandas as pd
import numpy as np

from tsfresh.utilities import dataframe_functions, profiling
from tsfresh.feature_extraction.settings import FeatureExtractionSettings

from itertools import groupby

_logger = logging.getLogger(__name__)


def extract_features(timeseries_container, feature_extraction_settings=None,
                     column_id=None, column_sort=None, column_kind=None, column_value=None,
                     parallelization=None):
    """
    Extract features from

    * a :class:`pandas.DataFrame` containing the different time series

    or

    * a dictionary of :class:`pandas.DataFrame` each containing one type of time series

    In both cases a :class:`pandas.DataFrame` with the calculated features will be returned.

    For a list of all the calculated time series features, please see the
    :class:`~tsfresh.feature_extraction.settings.FeatureExtractionSettings` class,
    which is used to control which features with which parameters are calculated.

    For a detailed explanation of the different parameters and data formats please see :ref:`data-formats-label`.

    Examples
    ========

    >>> from tsfresh.examples import load_robot_execution_failures
    >>> from tsfresh import extract_features
    >>> df, _ = load_robot_execution_failures()
    >>> X = extract_features(df, column_id='id', column_sort='time')

    which would give the same results as described above. In this case, the column_kind is not allowed.
    Except that, the same rules for leaving out the columns apply as above.

    :param timeseries_container: The pandas.DataFrame with the time series to compute the features for, or a
            dictionary of pandas.DataFrames.
    :type timeseries_container: pandas.DataFrame or dict

    :param feature_extraction_settings: settings object that controls which features are calculated
    :type feature_extraction_settings: tsfresh.feature_extraction.settings.FeatureExtractionSettings

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
    :type parallelization: str

    :return: The (maybe imputed) DataFrame containing extracted features.
    :rtype: pandas.DataFrame
    """
    # Always use the standardized way of storing the data.
    # See the function normalize_input_to_internal_representation for more information.
    kind_to_df_map, column_id, column_value = \
        dataframe_functions.normalize_input_to_internal_representation(timeseries_container, column_id, column_sort,
                                                                       column_kind, column_value)

    # Use the standard setting if the user did not supply ones himself.
    if feature_extraction_settings is None:
        feature_extraction_settings = FeatureExtractionSettings()
        for key in kind_to_df_map:
            feature_extraction_settings.set_default_parameters(key)

    # Choose the parallelization according to a rule-of-thumb
    if parallelization is None:
        parallelization = 'per_sample' if (feature_extraction_settings.n_processes / 2) > len(kind_to_df_map) \
            else 'per_kind'

    _logger.info('Parallelizing feature calculation {}'.format(parallelization))

    # If requested, do profiling (advanced feature)
    if feature_extraction_settings.PROFILING:
        profiler = profiling.start_profiling()

    # Calculate the result
    if parallelization == 'per_kind':
        result = _extract_features_parallel_per_kind(kind_to_df_map, feature_extraction_settings,
                                                     column_id, column_value)
    elif parallelization == 'per_sample':
        result = _extract_features_parallel_per_sample(kind_to_df_map, feature_extraction_settings,
                                                       column_id, column_value)
    else:
        raise ValueError("Argument parallelization must be one of: 'per_kind', 'per_sample'")

    # Impute the result if requested
    if feature_extraction_settings.IMPUTE is not None:
        feature_extraction_settings.IMPUTE(result)

    # Turn off profiling if it was turned on
    if feature_extraction_settings.PROFILING:
        profiling.end_profiling(profiler, filename=feature_extraction_settings.PROFILING_FILENAME,
                                sorting=feature_extraction_settings.PROFILING_SORTING)

    return result


def _extract_features_parallel_per_kind(kind_to_df_map, settings, column_id, column_value):
    """
    Parallelize the feature extraction per kind.

    :param kind_to_df_map: The time series to compute the features for in our internal format
    :type kind_to_df_map: dict of pandas.DataFrame

    :param column_id: The name of the id column to group by.
    :type column_id: str
    :param column_value: The name for the column keeping the value itself.
    :type column_value: str

    :param settings: settings object that controls which features are calculated
    :type settings: tsfresh.feature_extraction.settings.FeatureExtractionSettings

    :return: The (maybe imputed) DataFrame containing extracted features.
    :rtype: pandas.DataFrame
    """
    partial_extract_features_for_one_time_series = partial(_extract_features_for_one_time_series,
                                                           column_id=column_id,
                                                           column_value=column_value,
                                                           settings=settings)
    pool = Pool(settings.n_processes)

    extracted_features = pool.map(partial_extract_features_for_one_time_series, kind_to_df_map.items(),
                                  chunksize=settings.chunksize)

    pool.close()

    # Concatenate all partial results
    result = pd.concat((df for _, df in extracted_features), axis=1, join='outer').astype(np.float64)

    pool.join()
    return result


def _extract_features_parallel_per_sample(kind_to_df_map, settings, column_id, column_value):
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

    :param settings: settings object that controls which features are calculated
    :type settings: tsfresh.feature_extraction.settings.FeatureExtractionSettings

    :return: The (maybe imputed) DataFrame containing extracted features.
    :rtype: pandas.DataFrame
    """

    partial_extract_features_for_one_time_series = partial(_extract_features_for_one_time_series,
                                                           column_id=column_id,
                                                           column_value=column_value,
                                                           settings=settings)
    pool = Pool(settings.n_processes)

    # Submit map jobs per kind per sample in a generator object
    kind_and_df_group_list = ((kind, df_group)
                              for kind, df_kind in kind_to_df_map.items()
                              for _, df_group in df_kind.groupby(column_id))

    dfs_per_kind = pool.map(
        partial_extract_features_for_one_time_series,
        kind_and_df_group_list,
        chunksize=settings.chunksize,
    )

    # Wait for the pool to finish its job
    pool.close()
    pool.join()

    # dfs_per_kind is now a list of (kind, df). We want to extract all dfs with the same kind and concatenate them
    # We have to make sure to not create unnecessary copies here.

    # We will use groupby to group the list by their kind entry. The keyfunction extracts this kind for us.
    keyfunction = lambda (kind, df): kind
    dfs_per_kind = sorted(dfs_per_kind, key=keyfunction)
    dfs_per_kind = (pd.concat((df for kind, df in dfs), axis=0).astype(np.float64)
                    for kind, dfs in groupby(dfs_per_kind, keyfunction))

    result = pd.concat(dfs_per_kind, axis=1).astype(np.float64)

    return result


def _extract_features_for_one_time_series(prefix_and_dataframe, column_id, column_value, settings):
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
    passed settings instance (see :class:`~tsfresh.feature_extraction.settings.FeatureExtractionSettings` for a list of
    all possible features to calculate).

    The parameter `dataframe` is not allowed to have any NaN value in it. It is possible to have different numbers
    of values for different ids.

    :param prefix_and_dataframe: Tuple of column_prefix and dataframe
        column_prefix is the string that each extracted feature will be prefixed with (for better separation)
        dataframe with at least the columns column_id and column_value to extract the time
        series features for.
    :type prefix_and_dataframe: (str, DataFrame)
    :param column_id: The name of the column with the ids.
    :param column_value: The name of the column with the values.
    :param settings: The settings to control, which features will be extracted.
    :return: A dataframe with the extracted features as the columns (prefixed with column_prefix) and as many
        rows as their are unique values in the id column.
    """
    column_prefix, dataframe = prefix_and_dataframe
    column_prefix = str(column_prefix)

    # Ensure features are calculated on float64
    dataframe[column_value] = dataframe[column_value].astype(np.float64)

    with warnings.catch_warnings():
        if not settings.show_warnings:
            warnings.simplefilter("ignore")
        else:
            warnings.simplefilter("default")

        if settings.set_default and column_prefix not in settings.kind_to_calculation_settings_mapping:
            settings.set_default_parameters(column_prefix)

        # Calculate the aggregation functions
        column_name_to_aggregate_function = settings.get_aggregate_functions(column_prefix)

        if column_name_to_aggregate_function:
            extracted_features = dataframe.groupby(column_id)[column_value].aggregate(column_name_to_aggregate_function)
        else:
            extracted_features = pd.DataFrame(index=dataframe[column_id].unique())

        # Calculate the apply functions
        apply_functions = settings.get_apply_functions(column_prefix)

        if apply_functions:
            list_of_extracted_feature_dataframes = [extracted_features]
            for apply_function, kwargs in apply_functions:
                current_result = dataframe.groupby(column_id)[column_value].apply(apply_function, **kwargs).unstack()
                if len(current_result) > 0:
                    list_of_extracted_feature_dataframes.append(current_result)

            if len(list_of_extracted_feature_dataframes) > 0:
                extracted_features = pd.concat(list_of_extracted_feature_dataframes, axis=1,
                                               join_axes=[extracted_features.index])

        return column_prefix, extracted_features
