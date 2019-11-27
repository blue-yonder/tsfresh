# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
Utility functions for handling the DataFrame conversions to the internal normalized format
(see ``normalize_input_to_internal_representation``) or on how to handle ``NaN`` and ``inf`` in the DataFrames.
"""
import gc
import warnings

import numpy as np
import pandas as pd


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

    # Make the replacement dataframes as large as the real one
    col_to_max = pd.DataFrame([col_to_max] * len(df_impute), index=df_impute.index)
    col_to_min = pd.DataFrame([col_to_min] * len(df_impute), index=df_impute.index)
    col_to_median = pd.DataFrame([col_to_median] * len(df_impute), index=df_impute.index)

    df_impute.where(df_impute.values != np.PINF, other=col_to_max, inplace=True)
    df_impute.where(df_impute.values != np.NINF, other=col_to_min, inplace=True)
    df_impute.where(~np.isnan(df_impute.values), other=col_to_median, inplace=True)

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
    data = df.values
    masked = np.ma.masked_invalid(data)
    columns = df.columns

    is_col_non_finite = masked.mask.sum(axis=0) == masked.data.shape[0]

    if np.any(is_col_non_finite):
        # We have columns that does not contain any finite value at all, so we will store 0 instead.
        warnings.warn("The columns {} did not have any finite values. Filling with zeros.".format(
            df.iloc[:, np.where(is_col_non_finite)[0]].columns.values), RuntimeWarning)

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


def get_ids(df_or_dict, column_id):
    """
    Aggregates all ids in column_id from the time series container `

    :param df_or_dict: a pandas DataFrame or a dictionary.
    :type df_or_dict: pandas.DataFrame or dict
    :param column_id: it must be present in the pandas DataFrame or in all DataFrames in the dictionary.
        It is not allowed to have NaN values in this column.
    :type column_id: basestring

    :return: as set with all existing ids in energy_ratio_by_chunks
    :rtype: Set
    :raise: ``TypeError`` if df_or_dict is not of type dict or pandas.DataFrame
    """
    if isinstance(df_or_dict, pd.DataFrame):
        return set(df_or_dict[column_id])
    elif isinstance(df_or_dict, dict):
        return set.union(*[set(df[column_id]) for _, df in df_or_dict.items()])
    else:
        raise TypeError("df_or_dict should be of type dict or pandas.DataFrame")


# todo: add more testcases
# todo: rewrite in a more straightforward way
def _normalize_input_to_internal_representation(timeseries_container, column_id, column_sort,
                                                column_kind, column_value):
    """
    Try to transform any given input to the internal representation of time series, which is a flat DataFrame
    (the first format from see :ref:`data-formats-label`).

    This function can transform pandas DataFrames in different formats or dictionaries into the internal format
    that we use. It should not be called by the user.

    :param timeseries_container: a pandas DataFrame or a dictionary. The required shape/form of the object depends on
        the rest of the passed arguments.
    :type timeseries_container: pandas.DataFrame or dict
    :param column_id: it must be present in the pandas DataFrame or in all DataFrames in the dictionary.
        It is not allowed to have NaN values in this column.
    :type column_id: basestring
    :param column_sort: if not None, sort the rows by this column. It is not allowed to
        have NaN values in this column.
    :type column_sort: basestring or None
    :param column_kind: It can only be used when passing a pandas DataFrame (the dictionary is already assumed to be
        grouped by the kind). Is must be present in the DataFrame and no NaN values are allowed. The DataFrame
        will be grouped by the values in the kind column and each group will be one entry in the resulting
        mapping.
        If the kind column is not passed, it is assumed that each column in the pandas DataFrame (except the id or
        sort column) is a possible kind and the DataFrame is split up into as many DataFrames as there are columns.
        It is not allowed to have a value column then.
    :type column_kind: basestring or None
    :param column_value: If it is given, it must be present and not-NaN on the pandas DataFrames (or all pandas
        DataFrames in the dictionaries). If it is None, the kind column must also be none.
    :type column_value: basestring or None

    :return: A tuple of 4 elements: the normalized DataFrame, the name of the id column, the name of the value column
             and the name of the value column
    :rtype: (pd.DataFrame, basestring, basestring, basestring)
    :raise: ``ValueError`` when the passed combination of parameters is wrong or does not fit to the input DataFrame
            or dict.
    """
    # Also make it possible to have a dict as an input
    if isinstance(timeseries_container, dict):
        if column_kind is not None:
            raise ValueError("You passed in a dictionary and gave a column name for the kind. Both are not possible.")

        column_kind = "_variables"

        timeseries_container = {key: df.copy() for key, df in timeseries_container.items()}

        for kind, df in timeseries_container.items():
            df[column_kind] = kind

        try:
            timeseries_container = pd.concat(timeseries_container.values(), sort=True)
        except TypeError:  # pandas < 0.23.0
            timeseries_container = pd.concat(timeseries_container.values())
        gc.collect()

    # Check ID column
    if column_id is None:
        raise ValueError("You have to set the column_id which contains the ids of the different time series")

    if column_id not in timeseries_container.columns:
        raise AttributeError("The given column for the id is not present in the data.")

    if timeseries_container[column_id].isnull().any():
        raise ValueError("You have NaN values in your id column.")

    # Check sort column
    if column_sort is not None:
        if timeseries_container[column_sort].isnull().any():
            raise ValueError("You have NaN values in your sort column.")

    # Check that either kind and value is None or both not None.
    if column_kind is None and column_value is not None:
        column_kind = "_variables"
        timeseries_container = timeseries_container.copy()
        timeseries_container[column_kind] = column_value
    if column_kind is not None and column_value is None:
        raise ValueError("If passing the kind, you also have to pass the value.")

    if column_kind is None and column_value is None:
        if column_sort is not None:
            sort = timeseries_container[column_sort].values
            timeseries_container = timeseries_container.drop(column_sort, axis=1)
        else:
            sort = range(len(timeseries_container))
            column_sort = "_sort"

        column_kind = "_variables"
        column_value = "_values"

        if not set(timeseries_container.columns) - {column_id}:
            raise ValueError("There is no column with values in your data!")

        # We need to preserve the index. However, pandas has hard times to parse the columns if they
        # have different types, so we need to preserve them.
        # At least until https://github.com/pandas-dev/pandas/pull/28859 is merged.
        if isinstance(column_id, int) or isinstance(column_id, float):
            # some arbitrary number
            index_name = column_id + 999
        else:
            index_name = "_temporary_index_column"

        timeseries_container.index.name = index_name
        timeseries_container = pd.melt(timeseries_container.reset_index(),
                                       id_vars=[index_name, column_id],
                                       value_name=column_value, var_name=column_kind)
        timeseries_container = timeseries_container.set_index(index_name)
        timeseries_container[column_sort] = np.tile(sort, (len(timeseries_container) // len(sort)))

    # Check kind column
    if column_kind not in timeseries_container.columns:
        raise AttributeError("The given column for the kind is not present in the data.")

    if timeseries_container[column_kind].isnull().any():
        raise ValueError("You have NaN values in your kind column.")

    # Check value column
    if column_value not in timeseries_container.columns:
        raise ValueError("The given column for the value is not present in the data.")

    if timeseries_container[column_value].isnull().any():
        raise ValueError("You have NaN values in your value column.")

    if column_sort:
        timeseries_container = timeseries_container.sort_values([column_id, column_kind, column_sort])
        timeseries_container = timeseries_container.drop(column_sort, axis=1)
    else:
        timeseries_container = timeseries_container.sort_values([column_id, column_kind])

    # The kind columns should always be of type "str" to make the inference of feature settings later in `from_columns`
    # work
    timeseries_container[column_kind] = timeseries_container[column_kind].astype(str)

    # Make sure we have only parsable names
    for kind in timeseries_container[column_kind].unique():
        if kind.endswith("_"):
            raise ValueError("The kind {kind} is not allowed to end with '_'".format(kind=kind))
        if "__" in kind:
            raise ValueError("The kind {kind} is not allowed to contain '__'".format(kind=kind))

    return timeseries_container, column_id, column_kind, column_value


def roll_time_series(df_or_dict, column_id, column_sort, column_kind, rolling_direction, max_timeshift=None):
    """
    This method creates sub windows of the time series. It rolls the (sorted) data frames for each kind and each id
    separately in the "time" domain (which is represented by the sort order of the sort column given by `column_sort`).

    For each rolling step, a new id is created by the scheme "id={id}, shift={shift}", here id is the former id of the
    column and shift is the amount of "time" shifts.

    A few remarks:

     * This method will create new IDs!
     * The sign of rolling defines the direction of time rolling, a positive value means we are going back in time
     * It is possible to shift time series of different lengths but
     * We assume that the time series are uniformly sampled
     * For more information, please see :ref:`forecasting-label`.

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
    :param max_timeshift: If not None, shift only up to max_timeshift. If None, shift as often as possible.
    :type max_timeshift: int

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
                                      rolling_direction=rolling_direction,
                                      max_timeshift=max_timeshift)
                for key in df_or_dict}

    # Now we know that this is a pandas data frame
    df = df_or_dict

    if column_id is not None:
        if column_id not in df:
            raise AttributeError("The given column for the id is not present in the data.")
    else:
        raise ValueError("You have to set the column_id which contains the ids of the different time series")

    if column_kind is not None:
        grouper = [column_kind, column_id]
    else:
        grouper = [column_id, ]

    if column_sort is not None and df[column_sort].dtype != np.object:

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
    max_timeshift = max_timeshift or grouped_data.count().max().max()

    if np.isnan(max_timeshift):
        raise ValueError("Somehow the maximum length of your time series is NaN (Does your time series container have "
                         "only one row?). Can not perform rolling.")

    if rolling_direction > 0:
        range_of_shifts = range(max_timeshift, -1, -1)
    else:
        range_of_shifts = range(-max_timeshift, 1)

    # Todo: not default for columns_sort to be None
    if column_sort is None:
        column_sort = "sort"
        df[column_sort] = range(df.shape[0])

    def roll_out_time_series(time_shift):
        # Shift out only the first "time_shift" rows
        df_temp = grouped_data.shift(time_shift)
        df_temp[column_id] = df[column_sort]
        if column_kind:
            df_temp[column_kind] = df[column_kind]
        return df_temp.dropna()

    df_shift = pd.concat([roll_out_time_series(time_shift) for time_shift in range_of_shifts], ignore_index=True)

    return df_shift.sort_values(by=[column_id, column_sort])


def make_forecasting_frame(x, kind, max_timeshift, rolling_direction):
    """
    Takes a singular time series x and constructs a DataFrame df and target vector y that can be used for a time series
    forecasting task.

    The returned df will contain, for every time stamp in x, the last max_timeshift data points as a new
    time series, such can be used to fit a time series forecasting model.

    See :ref:`forecasting-label` for a detailed description of the rolling process and how the feature matrix and target
    vector are derived.

    The returned time series container df, will contain the rolled time series as a flat data frame, the first format
    from :ref:`data-formats-label`.

    When x is a pandas.Series, the index will be used as id.

    :param x: the singular time series
    :type x: np.array or pd.Series
    :param kind: the kind of the time series
    :type kind: str
    :param rolling_direction: The sign decides, if to roll backwards (if sign is positive) or forwards in "time"
    :type rolling_direction: int
    :param max_timeshift: If not None, shift only up to max_timeshift. If None, shift as often as possible.
    :type max_timeshift: int

    :return: time series container df, target vector y
    :rtype: (pd.DataFrame, pd.Series)
    """
    n = len(x)

    if isinstance(x, pd.Series):
        t = x.index
    else:
        t = range(n)

    df = pd.DataFrame({"id": ["id"] * n,
                       "time": t,
                       "value": x,
                       "kind": kind})

    df_shift = roll_time_series(df,
                                column_id="id",
                                column_sort="time",
                                column_kind="kind",
                                rolling_direction=rolling_direction,
                                max_timeshift=max_timeshift)

    # drop the rows which should actually be predicted
    def mask_first(x):
        """
        this mask returns an array of 1s where the last entry is a 0
        """
        result = np.ones(len(x))
        result[-1] = 0
        return result

    mask = df_shift.groupby(['id'])['id'].transform(mask_first).astype(bool)
    df_shift = df_shift[mask]

    return df_shift, df["value"][1:]
