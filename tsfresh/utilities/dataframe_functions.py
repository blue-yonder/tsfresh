# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
Utility functions for handling the DataFrame conversions to the internal normalized format
(see ``normalize_input_to_internal_representation``) or on how to handle ``NaN`` and ``inf`` in the DataFrames.
"""

import numpy as np
import pandas as pd
import logging


_logger = logging.getLogger(__name__)


def check_for_nans_in_columns(df, columns=None):
    """
    Helper function to check for ``NaN`` in the data frame and raise a ``ValueError`` if there is one.

    :param df: the pandas DataFrame to test for NaNs
    :type df: pandas.DataFrame
    :param columns: a list of columns to test for NaNs. If left empty, all columns of the DataFrame will be tested.
    :type columns: list

    :return: None
    :rtype: None
    :raise: ``ValueError`` of ``NaNs`` are found in the DataFrame.
    """
    if columns is None:
        columns = df.columns

    for key in columns:
        if df[key].isnull().any():
            raise ValueError("Column {} of dataframe must not contain NaN values".format(key))


def impute(df_impute):
    """
    Columnwise replaces all ``NaNs`` and ``infs`` from the DataFrame `df_impute` with average/extreme values from
    the same columns. This is done as follows: Each occurring ``inf`` or ``NaN`` in `df_impute` is replaced by

        * ``-inf`` -> ``min``
        * ``+inf`` -> ``max``
        * ``NaN`` -> ``median``

    If the column does not contain finite values at all, it is filled with zeros.

    This function modifies `df_impute` in place. After that, df_impute is
    guaranteed to not contain any non-finite values. Also, all columns will be guaranteed to be of type ``np.float64``.

    :param df_impute: DataFrame to impute
    :type df_impute: pandas.DataFrame
    :return: None
    :rtype: None
    """
    col_to_max, col_to_min, col_to_median = get_range_values_per_column(df_impute)
    impute_dataframe_range(df_impute, col_to_max, col_to_min, col_to_median)


def impute_dataframe_zero(df_impute):
    """
    Replaces all ``NaNs`` and ``infs`` from the DataFrame `df_impute` with 0s.

    `df_impute` will be modified in place. All its columns will be of datatype ``np.float64``.

    :param df_impute: DataFrame to impute
    :type df_impute: pandas.DataFrame
    """

    df_impute.replace([np.PINF, np.NINF], 0, inplace=True)
    df_impute.fillna(0, inplace=True)


def impute_dataframe_range(df_impute, col_to_max=None, col_to_min=None, col_to_median=None):
    """
    Columnwise replaces all ``NaNs`` and ``infs`` from the DataFrame `df_impute` with average/extreme values from
    the provided dictionaries. This is done as follows: Each occurring ``inf`` or ``NaN`` in `df_impute` is replaced by

        * ``-inf`` -> ``min``
        * ``+inf`` -> ``max``
        * ``NaN`` -> ``median``

    If a column is not found in the one of the dictionaries, the values are calculated from the columns finite values.
    If the column does not contain finite values at all, it is filled with zeros.

    This function modifies `df_impute` in place. Unless the dictionaries contain ``NaNs`` or ``infs``, df_impute is
    guaranteed to not contain any non-finite values. Also, all columns will be guaranteed to be of type ``np.float64``.

    :param df_impute: DataFrame to impute
    :type df_impute: pandas.DataFrame
    :param col_to_max: Dictionary mapping column names to max values
    :type col_to_max: dict
    :param col_to_min: Dictionary mapping column names to min values
    :type col_to_max: dict
    :param col_to_median: Dictionary mapping column names to median values
    :type col_to_max: dict
    """

    if col_to_median is None:
        col_to_median = {}
    if col_to_min is None:
        col_to_min = {}
    if col_to_max is None:
        col_to_max = {}
    for column_name in df_impute.columns:
        column = df_impute[column_name]

        # If we do not have all three values (max, min, median) we have to get them
        if not (column_name in col_to_max and column_name in col_to_median and column_name in col_to_min):
            finite_values_in_column = column[np.isfinite(column)]

            if len(finite_values_in_column) == 0:
                _logger.warning(
                    "The replacement column {} did not have any finite values. Filling with zeros.".format(column_name))
                df_impute[column_name] = [0] * len(column)
                continue

        if column_name not in col_to_max:
            col_to_max[column_name] = max(finite_values_in_column)

        if column_name not in col_to_min:
            col_to_min[column_name] = min(finite_values_in_column)

        if column_name not in col_to_median:
            col_to_median[column_name] = np.median(finite_values_in_column)

        # Finally, replace
        df_impute[column_name] = df_impute[column_name].replace(np.PINF, col_to_max[column_name]). \
            replace(np.NINF, col_to_min[column_name]). \
            fillna(col_to_median[column_name]). \
            astype(np.float64)

    return df_impute


def get_range_values_per_column(df):
    """
    Retrieves the finite max, min and mean values per column in `df` and stores them in three dictionaries, each mapping
    from column name to value. If a column does not contain finite value, 0 is stored instead.

    :param df: ``Dataframe``
    :return: Dictionaries mapping column names to max, min, mean values
    """
    col_to_max = {}
    col_to_min = {}
    col_to_median = {}

    for column_name in df.columns:
        column = df[column_name]
        finite_values_in_column = column[np.isfinite(column)]

        if len(finite_values_in_column) == 0:
            finite_values_in_column = [0]

        col_to_max[column_name] = max(finite_values_in_column)
        col_to_min[column_name] = min(finite_values_in_column)
        col_to_median[column_name] = np.median(finite_values_in_column)

    return col_to_max, col_to_min, col_to_median


def restrict_input_to_index(df_or_dict, column_id, index):
    """
    Restrict df_or_dict to those ids contained in index.

    :param df_or_dict: a pandas DataFrame or a dictionary.
    :type df_or_dict: pandas.DataFrame or dict
    :param column_id: it must be present in the pandas DataFrame or in all DataFrames in the dictionary.
        It is not allowed to have NaN values in this column.
    :type column_id: basestring
    :param index: Index containing the ids
    :type index: Iterable or pandas.Series
    :return: the restricted df_or_dict
    :rtype: dict or pandas.DataFrame
    :raise: ``TypeError`` if df_or_dict is not of type dict or pandas.DataFrame
    """
    if isinstance(df_or_dict, pd.DataFrame):
        df_or_dict_restricted = df_or_dict[df_or_dict[column_id].isin(index)]
    elif isinstance(df_or_dict, dict):
        df_or_dict_restricted = {kind: df[df[column_id].isin(index)]
                                  for kind, df in df_or_dict.items()}
    else:
        raise TypeError("df_or_dict should be of type dict or pandas.DataFrame")

    return df_or_dict_restricted


# todo: add more testcases
# todo: rewrite in a more straightforward way
def normalize_input_to_internal_representation(df_or_dict, column_id, column_sort, column_kind, column_value):
    """
    Try to transform any given input to the internal representation of time series, which is a mapping from string
    (the kind) to a pandas DataFrame with exactly two columns (the value and the id).

    This function can transform pandas DataFrames in different formats or dictionaries to pandas DataFrames in different
    formats. It is used internally in the extract_features function and should not be called by the user.

    :param df_or_dict: a pandas DataFrame or a dictionary. The required shape/form of the object depends on the rest of
        the passed arguments.
    :type df_or_dict: pandas.DataFrame or dict
    :param column_id: if not None, it must be present in the pandas DataFrame or in all DataFrames in the dictionary.
        It is not allowed to have NaN values in this column.
        If this column name is None, a new column will be added to the pandas DataFrame (or all pandas DataFrames in
        the dictionary) and the same id for all entries is assumed.
    :type column_id: basestring or None
    :param column_sort: if not None, sort the rows by this column. Then, the column is dropped. It is not allowed to
        have NaN values in this column.
    :type column_sort: basestring or None
    :param column_kind: It can only be used when passing a pandas DataFrame (the dictionary is already assumed to be
        grouped by the kind). Is must be present in the DataFrame and no NaN values are allowed. The DataFrame
        will be grouped by the values in the kind column and each grouped will be one entry in the resulting
        mapping.
        If the kind column is not passed, it is assumed that each column in the pandas DataFrame (except the id or
        sort column) is a possible kind and the DataFrame is split up into as many DataFrames as there are columns.
        Except when a value column is given: then it is assumed that there is only one column.
    :type column_kind: basestring or None
    :param column_value: If it is given, it must be present and not-NaN on the pandas DataFrames (or all pandas
        DataFrames in the dictionaries). If it is None, it is assumed that there is only a single remaining column
        in the DataFrame(s) (otherwise an exception is raised).
    :type column_value: basestring or None
    :return: A tuple of 3 elements: the normalized DataFrame as a dictionary mapping from the kind (as a string) to the
             corresponding DataFrame, the name of the id column and the name of the value column
    :rtype: (dict, basestring, basestring)
    :raise: ``ValueError`` when the passed combination of parameters is wrong or does not fit to the input DataFrame
            or dict.
    """
    if isinstance(df_or_dict, dict):
        if column_kind is not None:
            raise ValueError("You passed in a dictionary and gave a column name for the kind. Both is not possible.")
        kind_to_df_map = {key: df.copy() for key, df in df_or_dict.items()}
    else:
        if column_kind is not None:
            kind_to_df_map = {key: group.copy().drop(column_kind, axis=1) for key, group in df_or_dict.groupby(column_kind)}
        else:
            if column_value is not None:
                kind_to_df_map = {column_value: df_or_dict.copy()}
            else:
                id_and_sort_column = [_f for _f in [column_id, column_sort] if _f is not None]
                kind_to_df_map = {key: df_or_dict[[key] + id_and_sort_column].copy().rename(columns={key: "_value"})
                                  for key in df_or_dict.columns if key not in id_and_sort_column}

                #todo: is this the right check?
                if len(kind_to_df_map) < 1:
                    raise ValueError("You passed in a dataframe without a value column.")
                column_value = "_value"

    if column_sort is not None:
        for kind in kind_to_df_map:
            # Require no Nans in column
            if kind_to_df_map[kind][column_sort].isnull().any():
                raise ValueError("You have NaN values in your sort column.")
            kind_to_df_map[kind] = kind_to_df_map[kind].sort_values(column_sort).drop(column_sort, axis=1)

    if column_id is not None:
        for kind in kind_to_df_map:
            if column_id not in kind_to_df_map[kind].columns:
                raise AttributeError("The given column for the id is not present in the data.")
            elif kind_to_df_map[kind][column_id].isnull().any():
                raise ValueError("You have NaN values in your id column.")
    else:
        raise ValueError("You have to set the column_id which contains the ids of the different time series")

    # Either the column for the value must be given...
    if column_value is not None:
        for kind in kind_to_df_map:
            if column_value not in kind_to_df_map[kind].columns:
                raise ValueError("The given column for the value is not present in the data.")
    # or it is not allowed to have more than one column left (except id and sort)
    else:
        # But this column has to be the same always:
        remaining_columns = set(col for kind in kind_to_df_map for col in kind_to_df_map[kind].columns) - {column_id}

        if len(remaining_columns) > 1:
            raise ValueError("You did not give a column for values and we would have to choose between more than one.")
        elif len(remaining_columns) < 1:
            raise ValueError("You passed in a dataframe without a value column.")
        else:
            column_value = list(set(remaining_columns))[0]

        for kind in kind_to_df_map:
            if not column_value in kind_to_df_map[kind].columns:
                raise ValueError("You did not pass a column_value and there is not a single candidate in all data frames.")

    # Require no NaNs in value columns
    for kind in kind_to_df_map:
        if kind_to_df_map[kind][column_value].isnull().any():
                raise ValueError("You have NaN values in your value column.")

    return kind_to_df_map, column_id, column_value
