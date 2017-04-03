# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
Utility functions for handling the DataFrame conversions to the internal normalized format
(see ``normalize_input_to_internal_representation``) or on how to handle ``NaN`` and ``inf`` in the DataFrames.
"""
import warnings

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

    if pd.isnull(df.loc[:, columns]).any().any():
        if not isinstance(columns, list):
            columns = list(columns)
        raise ValueError("Columns {} of DataFrame must not contain NaN values".format(
            df.loc[:, columns].columns[pd.isnull(df.loc[:, columns]).sum() > 0].tolist()))


def impute(df_impute):
    """
    Columnwise replaces all ``NaNs`` and ``infs`` from the DataFrame `df_impute` with average/extreme values from
    the same columns. This is done as follows: Each occurring ``inf`` or ``NaN`` in `df_impute` is replaced by

        * ``-inf`` -> ``min``
        * ``+inf`` -> ``max``
        * ``NaN`` -> ``median``

    If the column does not contain finite values at all, it is filled with zeros.

    This function modifies `df_impute` in place. After that, df_impute is guaranteed to not contain any non-finite
    values. Also, all columns will be guaranteed to be of type ``np.float64``.

    :param df_impute: DataFrame to impute
    :type df_impute: pandas.DataFrame

    :return df_impute: imputed DataFrame
    :rtype df_impute: pandas.DataFrame
    """
    col_to_max, col_to_min, col_to_median = get_range_values_per_column(df_impute)
    df_impute = impute_dataframe_range(df_impute, col_to_max, col_to_min, col_to_median)

    # Ensure a type of "np.float64"
    df_impute.astype(np.float64, copy=False)
    return df_impute


def impute_dataframe_zero(df_impute):
    """
    Replaces all ``NaNs``, ``-infs`` and ``+infs`` from the DataFrame `df_impute` with 0s.
    The `df_impute` will be modified in place. All its columns will be into converted into dtype ``np.float64``.

    :param df_impute: DataFrame to impute
    :type df_impute: pandas.DataFrame

    :return df_impute: imputed DataFrame
    :rtype df_impute: pandas.DataFrame
    """

    df_impute.replace([np.PINF, np.NINF], 0, inplace=True)
    df_impute.fillna(0, inplace=True)

    # Ensure a type of "np.float64"
    df_impute.astype(np.float64, copy=False)
    return df_impute


def impute_dataframe_range(df_impute, col_to_max, col_to_min, col_to_median):
    """
    Columnwise replaces all ``NaNs``, ``-inf`` and ``+inf`` from the DataFrame `df_impute` with average/extreme values
    from the provided dictionaries.

    This is done as follows: Each occurring ``inf`` or ``NaN`` in `df_impute` is replaced by

        * ``-inf`` -> by value in col_to_min
        * ``+inf`` -> by value in col_to_max
        * ``NaN`` -> by value in col_to_median

    If a column of df_impute is not found in the one of the dictionaries, this method will raise a ValueError.
    Also, if one of the values to replace is not finite a ValueError is returned

    This function modifies `df_impute` in place. Afterwards df_impute is
    guaranteed to not contain any non-finite values.
    Also, all columns will be guaranteed to be of type ``np.float64``.

    :param df_impute: DataFrame to impute
    :type df_impute: pandas.DataFrame
    :param col_to_max: Dictionary mapping column names to max values
    :type col_to_max: dict
    :param col_to_min: Dictionary mapping column names to min values
    :type col_to_max: dict
    :param col_to_median: Dictionary mapping column names to median values
    :type col_to_max: dict

    :return df_impute: imputed DataFrame
    :rtype df_impute: pandas.DataFrame
    :raise ValueError: if a column of df_impute is missing in col_to_max, col_to_min or col_to_median or a value
                       to replace is non finite
    """
    columns = df_impute.columns

    # Making sure col_to_median, col_to_max and col_to_min have entries for every column
    if not set(columns) <= set(col_to_median.keys()) or \
            not set(columns) <= set(col_to_max.keys()) or \
            not set(columns) <= set(col_to_min.keys()):
        raise ValueError("Some of the dictionaries col_to_median, col_to_max, col_to_min contains more or less keys "
                         "than the column names in df")

    # check if there are non finite values for the replacement
    if np.any(~np.isfinite(list(col_to_median.values()))) or \
            np.any(~np.isfinite(list(col_to_min.values()))) or \
            np.any(~np.isfinite(list(col_to_max.values()))):
        raise ValueError("Some of the dictionaries col_to_median, col_to_max, col_to_min contains non finite values "
                         "to replace")

    # Replacing values
    # +inf -> max
    indices = np.nonzero(df_impute.values == np.PINF)
    if len(indices[0]) > 0:
        replacement = [col_to_max[columns[i]] for i in indices[1]]
        df_impute.iloc[indices] = replacement

    # -inf -> min
    indices = np.nonzero(df_impute.values == np.NINF)
    if len(indices[0]) > 0:
        replacement = [col_to_min[columns[i]] for i in indices[1]]
        df_impute.iloc[indices] = replacement

    # NaN -> median
    indices = np.nonzero(np.isnan(df_impute.values))
    if len(indices[0]) > 0:
        replacement = [col_to_median[columns[i]] for i in indices[1]]
        df_impute.iloc[indices] = replacement

    df_impute.astype(np.float64, copy=False)
    return df_impute


def get_range_values_per_column(df):
    """
    Retrieves the finite max, min and mean values per column in the DataFrame `df` and stores them in three
    dictionaries. Those dictionaries `col_to_max`, `col_to_min`, `col_to_median` map the columnname to the maximal,
    minimal or median value of that column.

    If a column does not contain any finite values at all, a 0 is stored instead.

    :param df: the Dataframe to get columnswise max, min and median from
    :type df: pandas.DataFrame

    :return: Dictionaries mapping column names to max, min, mean values
    :rtype: (dict, dict, dict)
    """
    data = df.get_values()
    masked = np.ma.masked_invalid(data)
    columns = df.columns

    is_col_non_finite = masked.mask.sum(axis=0) == masked.data.shape[0]

    if np.any(is_col_non_finite):
        # We have columns that does not contain any finite value at all, so we will store 0 instead.
        _logger.warning("The columns {} did not have any finite values. Filling with zeros.".format(
            df.iloc[:, np.where(is_col_non_finite)[0]].columns.values))

        masked.data[:, is_col_non_finite] = 0  # Set the values of the columns to 0
        masked.mask[:, is_col_non_finite] = False  # Remove the mask for this column

    # fetch max, min and median for all columns
    col_to_max = dict(zip(columns, np.max(masked, axis=0)))
    col_to_min = dict(zip(columns, np.min(masked, axis=0)))
    col_to_median = dict(zip(columns, np.ma.median(masked, axis=0)))

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

    :return df_or_dict_restricted: the restricted df_or_dict
    :rtype df_or_dict_restricted: dict or pandas.DataFrame
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
    :param column_id: it must be present in the pandas DataFrame or in all DataFrames in the dictionary.
        It is not allowed to have NaN values in this column.
    :type column_id: basestring or None
    :param column_sort: if not None, sort the rows by this column. It is not allowed to
        have NaN values in this column.
    :type column_sort: basestring or None
    :param column_kind: It can only be used when passing a pandas DataFrame (the dictionary is already assumed to be
        grouped by the kind). Is must be present in the DataFrame and no NaN values are allowed. The DataFrame
        will be grouped by the values in the kind column and each group will be one entry in the resulting
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
            raise ValueError("You passed in a dictionary and gave a column name for the kind. Both are not possible.")
        kind_to_df_map = {key: df.copy() for key, df in df_or_dict.items()}
    else:
        if column_kind is not None:
            kind_to_df_map = {key: group.copy().drop(column_kind, axis=1) for key, group in
                              df_or_dict.groupby(column_kind)}
        else:
            if column_value is not None:
                kind_to_df_map = {column_value: df_or_dict.copy()}
            else:
                id_and_sort_column = [_f for _f in [column_id, column_sort] if _f is not None]
                kind_to_df_map = {key: df_or_dict[[key] + id_and_sort_column].copy().rename(columns={key: "_value"})
                                  for key in df_or_dict.columns if key not in id_and_sort_column}

                # TODO: is this the right check?
                if len(kind_to_df_map) < 1:
                    raise ValueError("You passed in a dataframe without a value column.")
                column_value = "_value"

    if column_id is not None:
        for kind in kind_to_df_map:
            if column_id not in kind_to_df_map[kind].columns:
                raise AttributeError("The given column for the id is not present in the data.")
            elif kind_to_df_map[kind][column_id].isnull().any():
                raise ValueError("You have NaN values in your id column.")
    else:
        raise ValueError("You have to set the column_id which contains the ids of the different time series")

    for kind in kind_to_df_map:
        kind_to_df_map[kind].index.name = None

    if column_sort is not None:
        for kind in kind_to_df_map:
            # Require no Nans in column
            if kind_to_df_map[kind][column_sort].isnull().any():
                raise ValueError("You have NaN values in your sort column.")

            kind_to_df_map[kind] = kind_to_df_map[kind].sort_values(column_sort).drop(column_sort, axis=1)

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
                raise ValueError(
                    "You did not pass a column_value and there is not a single candidate in all data frames.")

    # Require no NaNs in value columns
    for kind in kind_to_df_map:
        if kind_to_df_map[kind][column_value].isnull().any():
            raise ValueError("You have NaN values in your value column.")

    return kind_to_df_map, column_id, column_value


def roll_time_series(df_or_dict, column_id, column_sort, column_kind, rolling_direction,
                     maximum_number_of_timeshifts=None):
    """
    Roll the (sorted) data frames for each kind and each id separately in the "time" domain
    (which is represented by the sort order of the sort column given by `column_sort`).

    For each rolling step, a new id is created by the scheme "id={id}, shift={shift}", here id is the former id of the
    column and shift is the amount of "time" shifts.

    A few remarks:

     * This method will create new IDs!
     * The sign of rolling defines the direction of time rolling, a positive value means we are going back in time
     * It is possible to shift time series of different lenghts but
     * We assume that the time series are uniformly sampled
     * For more information, please see :ref:`rolling-label`.

    :param df_or_dict: a pandas DataFrame or a dictionary. The required shape/form of the object depends on the rest of
        the passed arguments.
    :type df_or_dict: pandas.DataFrame or dict
    :param column_id: it must be present in the pandas DataFrame or in all DataFrames in the dictionary.
        It is not allowed to have NaN values in this column.
    :type column_id: basestring or None
    :param column_sort: if not None, sort the rows by this column. It is not allowed to
        have NaN values in this column.
    :type column_sort: basestring or None
    :param column_kind: It can only be used when passing a pandas DataFrame (the dictionary is already assumed to be
        grouped by the kind). Is must be present in the DataFrame and no NaN values are allowed.
        If the kind column is not passed, it is assumed that each column in the pandas DataFrame (except the id or
        sort column) is a possible kind.
    :type column_kind: basestring or None
    :param rolling_direction: The sign decides, if to roll backwards or forwards in "time"
    :type rolling_direction: int
    :param maximum_number_of_timeshifts: If not None, shift only up to maximum_number_of_timeshifts.
        If None, shift as often as possible.
    :type maximum_number_of_timeshifts: int

    :return: The rolled data frame or dictionary of data frames
    :rtype: the one from df_or_dict
    """

    if rolling_direction == 0:
        raise ValueError("Rolling direction of 0 is not possible")

    if isinstance(df_or_dict, dict):
        if column_kind is not None:
            raise ValueError("You passed in a dictionary and gave a column name for the kind. Both are not possible.")

        return {key: roll_time_series(df_or_dict=df_or_dict[key],
                                      column_id=column_id,
                                      column_sort=column_sort,
                                      column_kind=column_kind,
                                      rolling_direction=rolling_direction)
                for key in df_or_dict}

    # Now we know that this is a pandas data frame
    df = df_or_dict

    if column_id is not None:
        if column_id not in df:
                raise AttributeError("The given column for the id is not present in the data.")
    else:
        raise ValueError("You have to set the column_id which contains the ids of the different time series")

    if column_kind is not None:
        grouper = (column_kind, column_id)
    else:
        grouper = (column_id,)

    if column_sort is not None:
        # Require no Nans in column
        if df[column_sort].isnull().any():
            raise ValueError("You have NaN values in your sort column.")

        df = df.sort_values(column_sort)

        # if rolling is enabled, the data should be uniformly sampled in this column
        # Build the differences between consecutive time sort values

        differences = df.groupby(grouper)[column_sort].apply(
            lambda x: x.values[:-1] - x.values[1:])
        # Write all of them into one big list
        differences = sum(map(list, differences), [])
        # Test if all differences are the same
        if differences and min(differences) != max(differences):
            warnings.warn("Your time stamps are not uniformly sampled, which makes rolling "
                          "nonsensical in some domains.")

    # Roll the data frames if requested
    rolling_direction = np.sign(rolling_direction)

    grouped_data = df.groupby(grouper)
    maximum_number_of_timeshifts = maximum_number_of_timeshifts or grouped_data.count().max().max()

    if np.isnan(maximum_number_of_timeshifts):
        raise ValueError("Somehow the maximum length of your time series is NaN (Does your time series container have "
                         "only one row?). Can not perform rolling.")

    if rolling_direction > 0:
        range_of_shifts = range(maximum_number_of_timeshifts, -1, -1)
    else:
        range_of_shifts = range(-maximum_number_of_timeshifts, 1)

    def roll_out_time_series(time_shift):
        # Shift out only the first "time_shift" rows
        df_temp = grouped_data.shift(time_shift)
        df_temp[column_id] = "id=" + df[column_id].map(str) + ", shift={}".format(time_shift)
        if column_kind:
            df_temp[column_kind] = df[column_kind]
        return df_temp.dropna()

    return pd.concat([roll_out_time_series(time_shift) for time_shift in range_of_shifts],
                     ignore_index=True)



