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
import pandas as pd
import numpy as np
from tsfresh.utilities import dataframe_functions, profiling
from tsfresh.feature_extraction.settings import FeatureExtractionSettings


def extract_features(timeseries_container, feature_extraction_settings=None,
                     column_id=None, column_sort=None, column_kind=None, column_value=None):
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

    :param column_id: The name of the id column to group by.
    :type column_id: str
    :param column_sort: The name of the sort column.
    :type column_sort: str
    :param column_kind: The name of the column keeping record on the kind of the value.
    :type column_kind: str
    :param column_value: The name for the column keeping the value itself.
    :type column_value: str

    :param feature_extraction_settings: settings object that controls which features are calculated
    :type feature_extraction_settings: tsfresh.feature_extraction.settings.FeatureExtractionSettings

    :return: The (maybe imputed) DataFrame with the extracted features.
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

    # If requested, do profiling (advanced feature)
    if feature_extraction_settings.PROFILING:
        profiler = profiling.start_profiling()

    # Extract the time series features for every type of time series and concatenate them together.
    all_possible_unique_id_values = set(id_value for kind, df in kind_to_df_map.items()
                                        for id_value in df[column_id])
    df_with_ids = pd.DataFrame(index=all_possible_unique_id_values)

    pool = Pool(feature_extraction_settings.n_processes)
    partial_extract_features_for_one_time_series = partial(_extract_features_for_one_time_series, column_id=column_id,
                              column_value=column_value, settings=feature_extraction_settings)
    extracted_features = pool.map(partial_extract_features_for_one_time_series, kind_to_df_map.items())

    # Add time series features to result
    result = pd.concat([df_with_ids] + extracted_features, axis=1, join='outer', join_axes=[df_with_ids.index])\
        .astype(np.float64)

    # Impute the result if requested
    if feature_extraction_settings.IMPUTE is not None:
        feature_extraction_settings.IMPUTE(result)

    # Turn off profiling if it was turned on
    if feature_extraction_settings.PROFILING:
        profiling.end_profiling(profiler, filename=feature_extraction_settings.PROFILING_FILENAME,
                                sorting=feature_extraction_settings.PROFILING_SORTING)

    pool.close()
    pool.join()

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

        return extracted_features
