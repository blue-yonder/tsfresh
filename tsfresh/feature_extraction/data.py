import itertools
from collections import namedtuple
from typing import Iterable, Sized

import pandas as pd


class Timeseries(namedtuple('Timeseries', ['id', 'kind', 'data'])):
    """
    Timeseries tuple used for feature extraction.

    Make sure `kind` is of type `str` to allow inference
    of feature settings in `feature_extraction.settings.from_columns`.
    """


class TsData(Iterable[Timeseries], Sized):
    """
    TsData provides access to time series data for internal usage.

    Implementations must implement `__iter__` and `__len__`.
    Other methods should be overwritten if a more efficient solution exists for the underlying data store.
    """

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SliceableTsData(TsData):
    """
    SliceableTsData uses a slice strategy to implement data partitioning.
    Implementations of `TsData` may extend this class if they can iterate their elements in deterministic order and
    if an efficient strategy exists to slice the data.

    Because `__iter__` defaults to `slice(0)`, implementations must only implement `slice`.
    """
    def __iter__(self):
        raise NotImplementedError


def _check_colname(*columns):
    """
    Check if given column names conflict with `settings.from_columns` (ends with '_' or contains '__').

    :param columns: the column names to check
    :type columns: str

    :return: None
    :rtype: None
    :raise: ``ValueError`` if column names are invalid.
    """

    for col in columns:
        if str(col).endswith("_"):
            raise ValueError("Dict keys are not allowed to end with '_': {}".format(col))

        if "__" in str(col):
            raise ValueError("Dict keys are not allowed to contain '__': {}".format(col))


def _check_nan(df, *columns):
    """
    Raise a ``ValueError`` if one of the columns does not exist or contains NaNs.

    :param df: the pandas DataFrame to test for NaNs
    :type df: pandas.DataFrame
    :param columns: a list of columns to test for NaNs. If left empty, all columns of the DataFrame will be tested.
    :type columns: str

    :return: None
    :rtype: None
    :raise: ``ValueError`` if ``NaNs`` are found in the DataFrame.
    """

    for col in columns:
        if col not in df.columns:
            raise ValueError("Column not found: {}".format(col))

        if df[col].isnull().any():
            raise ValueError("Column must not contain NaN values: {}".format(col))


def _get_value_columns(df, *other_columns):
    value_columns = [col for col in df.columns if col not in other_columns]

    if len(value_columns) == 0:
        raise ValueError("Could not guess the value column! Please hand it to the function as an argument.")

    return value_columns


class WideTsFrameAdapter(SliceableTsData):
    def __init__(self, df, column_id, column_sort=None, value_columns=None):
        """
        Adapter for Pandas DataFrames in wide format, where multiple columns contain different time series for
        the same id.

        :param df: the data frame
        :type df: pd.DataFrame

        :param column_id: the name of the column containing time series group ids
        :type column_id: str

        :param column_sort: the name of the column to sort on
        :type column_sort: str|None

        :param value_columns: list of column names to treat as time series values.
            If `None` or empty, all columns except `column_id` and `column_sort` will be used.
        :type value_columns: list[str]|None
        """
        if column_id is None:
            raise ValueError("A value for column_id needs to be supplied")

        _check_nan(df, column_id)

        if not value_columns:
            value_columns = _get_value_columns(df, column_id, column_sort)

        _check_nan(df, *value_columns)
        _check_colname(*value_columns)

        self.value_columns = value_columns

        if column_sort is not None:
            _check_nan(df, column_sort)
            self.df_grouped = df.sort_values([column_sort]).groupby([column_id])
        else:
            self.df_grouped = df.groupby([column_id])

    def __len__(self):
        return self.df_grouped.ngroups * len(self.value_columns)

    def __iter__(self):
        for group_name, group in self.df_grouped:
            for kind in self.value_columns:
                yield Timeseries(group_name, kind, group[kind])


class LongTsFrameAdapter(TsData):
    def __init__(self, df, column_id, column_kind, column_value=None, column_sort=None):
        """
        Adapter for Pandas DataFrames in long format, where different time series for the same id are
        labeled by column `column_kind`.

        :param df: the data frame
        :type df: pd.DataFrame

        :param column_id: the name of the column containing time series group ids
        :type column_id: str

        :param column_kind: the name of the column containing time series kinds for each id
        :type column_kind: str

        :param column_value: None or the name of the column containing time series values
            If `None`, try to guess it from the remaining, unused columns.
        :type column_value: str|None

        :param column_sort: the name of the column to sort on
        :type column_sort: str|None
        """
        if column_id is None:
            raise ValueError("A value for column_id needs to be supplied")
        if column_kind is None:
            raise ValueError("A value for column_kind needs to be supplied")

        if column_value is None:
            possible_value_columns = _get_value_columns(df, column_id, column_sort, column_kind)
            if len(possible_value_columns) != 1:
                raise ValueError("Could not guess the value column, as the number of unused columns os not equal to 1."
                                 f"These columns where currently unused: {','.join(possible_value_columns)}"
                                 "Please hand it to the function as an argument.")
            self.column_value = possible_value_columns[0]
        else:
            self.column_value = column_value

        _check_nan(df, column_id, column_kind, self.column_value)
        _check_colname(column_kind)

        if column_sort is not None:
            _check_nan(df, column_sort)
            self.df_grouped = df.sort_values([column_sort]).groupby([column_id, column_kind])
        else:
            self.df_grouped = df.groupby([column_id, column_kind])

    def __len__(self):
        return len(self.df_grouped)

    def __iter__(self):
        for group_key, group in self.df_grouped:
            yield Timeseries(group_key[0], str(group_key[1]), group[self.column_value])


class TsDictAdapter(TsData):
    def __init__(self, ts_dict, column_id, column_value, column_sort=None):
        """
        Adapter for a dict, which maps different time series kinds to Pandas DataFrames.

        :param ts_dict: a dict of data frames
        :type ts_dict: dict[str, pd.DataFrame]

        :param column_id: the name of the column containing time series group ids
        :type column_id: str

        :param column_value: the name of the column containing time series values
        :type column_value: str

        :param column_sort: the name of the column to sort on
        :type column_sort: str|None
        """

        _check_colname(*list(ts_dict.keys()))
        for df in ts_dict.values():
            _check_nan(df, column_id, column_value)

        self.column_value = column_value

        if column_sort is not None:
            for key, df in ts_dict.items():
                _check_nan(df, column_sort)

            self.grouped_dict = {key: df.sort_values([column_sort]).groupby(column_id)
                                 for key, df in ts_dict.items()}
        else:
            self.grouped_dict = {key: df.groupby(column_id) for key, df in ts_dict.items()}

    def __iter__(self):
        for kind, grouped_df in self.grouped_dict.items():
            for ts_id, group in grouped_df:
                yield Timeseries(ts_id, str(kind), group[self.column_value])

    def __len__(self):
        return sum(grouped_df.ngroups for grouped_df in self.grouped_dict.values())


def to_tsdata(df, column_id=None, column_kind=None, column_value=None, column_sort=None):
    """
    Wrap supported data formats as a TsData object, i.e. an iterable of individual time series.

    E.g. the DataFrame

        ====  ======  =========
          id  kind          val
        ====  ======  =========
           1  a       -0.21761
           1  a       -0.613667
           1  a       -2.07339
           2  b       -0.576254
           2  b       -1.21924
        ====  ======  =========

    into

        Iterable((1, 'a', pd.Series([-0.217610, -0.613667, -2.073386]),
         (2, 'b', pd.Series([-0.576254, -1.219238]))

    :param df: one of the supported input formats
    :type df: pd.DataFrame|dict|TsData

    :param column_id: The name of the id column to group by.
    :type column_id: str|None

    :param column_kind: The name of the column keeping record on the kind of the value.
    :type column_kind: str|None

    :param column_value: The name for the column keeping the value itself.
    :type column_value: str|None

    :param column_sort: The name for the column to sort on.
    :type column_sort: str|None

    :return: a data adapter
    :rtype: TsData
    """

    if isinstance(df, TsData):
        return df

    elif isinstance(df, pd.DataFrame):
        if column_kind is not None:
            return LongTsFrameAdapter(df, column_id, column_kind, column_value, column_sort)
        else:
            if column_value is not None:
                return WideTsFrameAdapter(df, column_id, column_sort, [column_value])
            else:
                return WideTsFrameAdapter(df, column_id, column_sort)

    elif isinstance(df, dict):
        return TsDictAdapter(df, column_id, column_value, column_sort)

    else:
        raise ValueError("df must be a DataFrame or a dict of DataFrames. "
                         "See https://tsfresh.readthedocs.io/en/latest/text/data_formats.html")
