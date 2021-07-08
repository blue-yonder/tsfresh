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

from tsfresh import defaults
from tsfresh.utilities.distribution import (
    DistributorBaseClass,
    MapDistributor,
    MultiprocessingDistributor,
)


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
        raise ValueError(
            "Columns {} of DataFrame must not contain NaN values".format(
                df.loc[:, columns]
                .columns[pd.isnull(df.loc[:, columns]).sum() > 0]
                .tolist()
            )
        )


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
    if len(df_impute) == 0:
        return df_impute

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
    if len(df_impute) == 0:
        return df_impute

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
    if len(df_impute) == 0:
        return df_impute

    columns = df_impute.columns

    # Making sure col_to_median, col_to_max and col_to_min have entries for every column
    if (
        not set(columns) <= set(col_to_median.keys())
        or not set(columns) <= set(col_to_max.keys())
        or not set(columns) <= set(col_to_min.keys())
    ):
        raise ValueError(
            "Some of the dictionaries col_to_median, col_to_max, col_to_min contains more or less keys "
            "than the column names in df"
        )

    # check if there are non finite values for the replacement
    if (
        np.any(~np.isfinite(list(col_to_median.values())))
        or np.any(~np.isfinite(list(col_to_min.values())))
        or np.any(~np.isfinite(list(col_to_max.values())))
    ):
        raise ValueError(
            "Some of the dictionaries col_to_median, col_to_max, col_to_min contains non finite values "
            "to replace"
        )

    # Make the replacement dataframes as large as the real one
    col_to_max = pd.DataFrame([col_to_max] * len(df_impute), index=df_impute.index)
    col_to_min = pd.DataFrame([col_to_min] * len(df_impute), index=df_impute.index)
    col_to_median = pd.DataFrame(
        [col_to_median] * len(df_impute), index=df_impute.index
    )

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
        warnings.warn(
            "The columns {} did not have any finite values. Filling with zeros.".format(
                df.iloc[:, np.where(is_col_non_finite)[0]].columns.values
            ),
            RuntimeWarning,
        )

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
        ids_in_df = set(df_or_dict[column_id])
        ids_in_index = set(index)
        present_ids = ids_in_index & ids_in_df
        if not present_ids:
            msg = "The ids of the time series container and the index of the input data X do not share any identifier!"
            raise AttributeError(msg)

        df_or_dict_restricted = df_or_dict[df_or_dict[column_id].isin(index)]
    elif isinstance(df_or_dict, dict):
        df_or_dict_restricted = {
            kind: restrict_input_to_index(df, column_id, index)
            for kind, df in df_or_dict.items()
        }
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


def _roll_out_time_series(
    timeshift,
    grouped_data,
    rolling_direction,
    max_timeshift,
    min_timeshift,
    column_sort,
    column_id,
):
    """
    Internal helper function for roll_time_series.
    This function has the task to extract the rolled forecast data frame of the number `timeshift`.
    This means it has shifted a virtual window if size `max_timeshift` (or infinite)
    `timeshift` times in the positive direction (for positive `rolling_direction`) or in negative direction
    (for negative `rolling_direction`).
    It starts counting from the first data point for each id (and kind) (or the last one for negative
    `rolling_direction`).
    The rolling happens for each `id` and `kind` separately.
    Extracted data smaller than `min_timeshift` + 1 are removed.

    Implementation note:
    Even though negative rolling direction means, we let the window shift in negative direction over the data,
    the counting of `timeshift` still happens from the first row onwards. Example:

        1   2   3   4

    If we do positive rolling, we extract the sub time series

      [ 1 ]               input parameter: timeshift=1, new id: ([id=]X,[timeshift=]1)
      [ 1   2 ]           input parameter: timeshift=2, new id: ([id=]X,[timeshift=]2)
      [ 1   2   3 ]       input parameter: timeshift=3, new id: ([id=]X,[timeshift=]3)
      [ 1   2   3   4 ]   input parameter: timeshift=4, new id: ([id=]X,[timeshift=]4)

    If we do negative rolling:

      [ 1   2   3   4 ]   input parameter: timeshift=1, new id: ([id=]X,[timeshift=]1)
          [ 2   3   4 ]   input parameter: timeshift=2, new id: ([id=]X,[timeshift=]2)
              [ 3   4 ]   input parameter: timeshift=3, new id: ([id=]X,[timeshift=]3)
                  [ 4 ]   input parameter: timeshift=4, new id: ([id=]X,[timeshift=]4)

    If you now reverse the order of the negative examples, it looks like shifting the
    window from the back (but it is implemented to start counting from the beginning).

    """

    def _f(x):
        if rolling_direction > 0:
            # For positive rolling, the right side of the window moves with `timeshift`
            shift_until = timeshift
            shift_from = max(shift_until - max_timeshift - 1, 0)

            df_temp = x.iloc[shift_from:shift_until] if shift_until <= len(x) else None
        else:
            # For negative rolling, the left side of the window moves with `timeshift`
            shift_from = max(timeshift - 1, 0)
            shift_until = shift_from + max_timeshift + 1

            df_temp = x.iloc[shift_from:shift_until]

        if df_temp is None or len(df_temp) < min_timeshift + 1:
            return

        df_temp = df_temp.copy()

        # and set the shift correctly
        if column_sort and rolling_direction > 0:
            timeshift_value = df_temp[column_sort].iloc[-1]
        elif column_sort and rolling_direction < 0:
            timeshift_value = df_temp[column_sort].iloc[0]
        else:
            timeshift_value = timeshift - 1
        # and now create new ones ids out of the old ones
        df_temp["id"] = df_temp[column_id].apply(lambda row: (row, timeshift_value))

        return df_temp

    return [grouped_data.apply(_f)]


def roll_time_series(
    df_or_dict,
    column_id,
    column_sort=None,
    column_kind=None,
    rolling_direction=1,
    max_timeshift=None,
    min_timeshift=0,
    chunksize=defaults.CHUNKSIZE,
    n_jobs=defaults.N_PROCESSES,
    show_warnings=defaults.SHOW_WARNINGS,
    disable_progressbar=defaults.DISABLE_PROGRESSBAR,
    distributor=None,
):
    """
    This method creates sub windows of the time series. It rolls the (sorted) data frames for each kind and each id
    separately in the "time" domain (which is represented by the sort order of the sort column given by `column_sort`).

    For each rolling step, a new id is created by the scheme ({id}, {shift}), here id is the former id of
    the column and shift is the amount of "time" shifts.
    You can think of it as having a window of fixed length (the max_timeshift) moving one step at a time over
    your time series.
    Each cut-out seen by the window is a new time series with a new identifier.

    A few remarks:

     * This method will create new IDs!
     * The sign of rolling defines the direction of time rolling, a positive value means we are shifting
       the cut-out window foreward in time. The name of each new sub time series is given by the last time point.
       This means, the time series named `([id=]4,[timeshift=]5)` with a `max_timeshift` of 3 includes the data
       of the times 3, 4 and 5.
       A negative rolling direction means, you go in negative time direction over your data.
       The time series named `([id=]4,[timeshift=]5)` with `max_timeshift` of 3 would then include the data
       of the times 5, 6 and 7.
       The absolute value defines how much time to shift at each step.
     * It is possible to shift time series of different lengths, but:
     * We assume that the time series are uniformly sampled
     * For more information, please see :ref:`forecasting-label`.

    :param df_or_dict: a pandas DataFrame or a dictionary. The required shape/form of the object depends on the rest of
        the passed arguments.
    :type df_or_dict: pandas.DataFrame or dict

    :param column_id: it must be present in the pandas DataFrame or in all DataFrames in the dictionary.
        It is not allowed to have NaN values in this column.
    :type column_id: basestring

    :param column_sort: if not None, sort the rows by this column. It is not allowed to
        have NaN values in this column. If not given, will be filled by an increasing number,
        meaning that the order of the passed dataframes are used as "time" for the time series.
    :type column_sort: basestring or None

    :param column_kind: It can only be used when passing a pandas DataFrame (the dictionary is already assumed to be
        grouped by the kind). Is must be present in the DataFrame and no NaN values are allowed.
        If the kind column is not passed, it is assumed that each column in the pandas DataFrame (except the id or
        sort column) is a possible kind.
    :type column_kind: basestring or None

    :param rolling_direction: The sign decides, if to shift our cut-out window backwards or forwards in "time".
        The absolute value decides, how much to shift at each step.
    :type rolling_direction: int

    :param max_timeshift: If not None, the cut-out window is at maximum `max_timeshift` large. If none, it grows
         infinitely.
    :type max_timeshift: int

    :param min_timeshift: Throw away all extracted forecast windows smaller or equal than this. Must be larger
         than or equal 0.
    :type min_timeshift: int

    :param n_jobs: The number of processes to use for parallelization. If zero, no parallelization is used.
    :type n_jobs: int

    :param chunksize: How many shifts per job should be calculated.
    :type chunksize: None or int

    :param show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
    :type show_warnings: bool

    :param disable_progressbar: Do not show a progressbar while doing the calculation.
    :type disable_progressbar: bool

    :param distributor: Advanced parameter: set this to a class name that you want to use as a
             distributor. See the utilities/distribution.py for more information. Leave to None, if you want
             TSFresh to choose the best distributor.
    :type distributor: class

    :return: The rolled data frame or dictionary of data frames
    :rtype: the one from df_or_dict
    """

    if rolling_direction == 0:
        raise ValueError("Rolling direction of 0 is not possible")

    if max_timeshift is not None and max_timeshift <= 0:
        raise ValueError("max_timeshift needs to be positive!")

    if min_timeshift < 0:
        raise ValueError("min_timeshift needs to be positive or zero!")

    if isinstance(df_or_dict, dict):
        if column_kind is not None:
            raise ValueError(
                "You passed in a dictionary and gave a column name for the kind. Both are not possible."
            )

        return {
            key: roll_time_series(
                df_or_dict=df_or_dict[key],
                column_id=column_id,
                column_sort=column_sort,
                column_kind=column_kind,
                rolling_direction=rolling_direction,
                max_timeshift=max_timeshift,
                min_timeshift=min_timeshift,
                chunksize=chunksize,
                n_jobs=n_jobs,
                show_warnings=show_warnings,
                disable_progressbar=disable_progressbar,
                distributor=distributor,
            )
            for key in df_or_dict
        }

    # Now we know that this is a pandas data frame
    df = df_or_dict

    if len(df) <= 1:
        raise ValueError(
            "Your time series container has zero or one rows!. Can not perform rolling."
        )

    if column_id is not None:
        if column_id not in df:
            raise AttributeError(
                "The given column for the id is not present in the data."
            )
    else:
        raise ValueError(
            "You have to set the column_id which contains the ids of the different time series"
        )

    if column_kind is not None:
        grouper = [column_kind, column_id]
    else:
        grouper = [
            column_id,
        ]

    if column_sort is not None:
        # Require no Nans in column
        if df[column_sort].isnull().any():
            raise ValueError("You have NaN values in your sort column.")

        df = df.sort_values(column_sort)

        if df[column_sort].dtype != np.object:
            # if rolling is enabled, the data should be uniformly sampled in this column
            # Build the differences between consecutive time sort values

            differences = df.groupby(grouper)[column_sort].apply(
                lambda x: x.values[:-1] - x.values[1:]
            )
            # Write all of them into one big list
            differences = sum(map(list, differences), [])
            # Test if all differences are the same
            if differences and min(differences) != max(differences):
                warnings.warn(
                    "Your time stamps are not uniformly sampled, which makes rolling "
                    "nonsensical in some domains."
                )

    # Roll the data frames if requested
    rolling_amount = np.abs(rolling_direction)
    rolling_direction = np.sign(rolling_direction)

    grouped_data = df.groupby(grouper)
    prediction_steps = grouped_data.count().max().max()

    max_timeshift = max_timeshift or prediction_steps

    # Todo: not default for columns_sort to be None
    if column_sort is None:
        df["sort"] = range(df.shape[0])

    if rolling_direction > 0:
        range_of_shifts = list(reversed(range(prediction_steps, 0, -rolling_amount)))
    else:
        range_of_shifts = range(1, prediction_steps + 1, rolling_amount)

    if distributor is None:
        if n_jobs == 0 or n_jobs == 1:
            distributor = MapDistributor(
                disable_progressbar=disable_progressbar, progressbar_title="Rolling"
            )
        else:
            distributor = MultiprocessingDistributor(
                n_workers=n_jobs,
                disable_progressbar=disable_progressbar,
                progressbar_title="Rolling",
                show_warnings=show_warnings,
            )

    if not isinstance(distributor, DistributorBaseClass):
        raise ValueError("the passed distributor is not an DistributorBaseClass object")

    kwargs = {
        "grouped_data": grouped_data,
        "rolling_direction": rolling_direction,
        "max_timeshift": max_timeshift,
        "min_timeshift": min_timeshift,
        "column_sort": column_sort,
        "column_id": column_id,
    }

    shifted_chunks = distributor.map_reduce(
        _roll_out_time_series,
        data=range_of_shifts,
        chunk_size=chunksize,
        function_kwargs=kwargs,
    )

    distributor.close()

    df_shift = pd.concat(shifted_chunks, ignore_index=True)

    return df_shift.sort_values(by=["id", column_sort or "sort"])


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

    df = pd.DataFrame({"id": ["id"] * n, "time": t, "value": x, "kind": kind})

    df_shift = roll_time_series(
        df,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        rolling_direction=rolling_direction,
        max_timeshift=max_timeshift,
    )

    # drop the rows which should actually be predicted
    def mask_first(x):
        """
        this mask returns an array of 1s where the last entry is a 0
        """
        result = np.ones(len(x))
        result[-1] = 0
        return result

    mask = df_shift.groupby(["id"])["id"].transform(mask_first).astype(bool)
    df_shift = df_shift[mask]

    # Now create the target vector out of the values
    # of the input series - not including the first one
    # (as there is nothing to forecast from)
    y = df["value"][1:]

    # make sure that the format is the same as the
    # df_shift index
    y.index = map(lambda x: ("id", x), y.index)

    return df_shift, y


def add_sub_time_series_index(
    df_or_dict, sub_length, column_id=None, column_sort=None, column_kind=None
):
    """
    Add a column "id" which contains:

    - if column_id is None: for each kind (or if column_kind is None for the full dataframe) a new index built by
      "sub-packaging" the data in packages of length "sub_length". For example if you have data with the
      length of 11 and sub_length is 2, you will get 6 new packages: 0, 0; 1, 1; 2, 2; 3, 3; 4, 4; 5.
    - if column_id is not None: the same as before, just for each id separately. The old column_id values are added
      to the new "id" column after a comma

    You can use this functions to turn a long measurement into sub-packages, where you want to extract features on.

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

    :return: The data frame or dictionary of data frames with a column "id" added
    :rtype: the one from df_or_dict
    """

    if isinstance(df_or_dict, dict):
        if column_kind is not None:
            raise ValueError(
                "You passed in a dictionary and gave a column name for the kind. Both are not possible."
            )

        return {
            key: add_sub_time_series_index(
                df_or_dict=df_or_dict[key],
                sub_length=sub_length,
                column_id=column_id,
                column_sort=column_sort,
                column_kind=column_kind,
            )
            for key in df_or_dict
        }

    df = df_or_dict

    grouper = []

    if column_id is not None:
        grouper.append(column_id)
    if column_kind is not None:
        grouper.append(column_kind)

    def _add_id_column(df_chunk):
        chunk_length = len(df_chunk)
        last_chunk_number = chunk_length // sub_length
        reminder = chunk_length % sub_length

        indices = np.concatenate(
            [
                np.repeat(np.arange(last_chunk_number), sub_length),
                np.repeat(last_chunk_number, reminder),
            ]
        )
        assert len(indices) == chunk_length

        if column_id:
            indices = list(zip(indices, df_chunk[column_id]))

        if column_sort:
            df_chunk = df_chunk.sort_values(column_sort)

        df_chunk["id"] = indices

        return df_chunk

    if grouper:
        df = df.groupby(grouper).apply(_add_id_column)
    else:
        df = _add_id_column(df)

    if column_sort:
        df = df.sort_values(column_sort)

    df = df.set_index(df.index.get_level_values(-1))

    return df
