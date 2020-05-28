import itertools
from typing import Iterable

import pandas as pd


class TsData(Iterable):
    """
    TsData implementations provide access to time series tuples
    """

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return sum(1 for _ in self)

    def slice(self, offset, length=None):
        """
        Get an iterable which iterates the data with given length from given offset

        :param offset: the position of the first element to return
        :type offset int

        :param length: the maximum number of elements to iterate
        :type length int|None

        :return: Iterable
        """
        end = offset + length if length is not None else None
        return itertools.islice(self, offset, end)

    def partition(self, chunk_size):
        """
        Split the data into iterable chunks of given chunk_size

        :param chunk_size:
        :type chunk_size int

        :return: Generator of iterables
        """
        x, y = divmod(len(self), chunk_size)
        chunk_info = [(i * chunk_size, chunk_size) for i in range(x)]
        if y > 0:
            chunk_info += [(x * chunk_size, y)]

        for offset, length in chunk_info:
            yield self.slice(offset, length)


class WideTsFrameAdapter(TsData):

    def __init__(self, df, column_id, column_sort=None, value_columns=None):
        """
        Adapter for Pandas DataFrames in wide format, where multiple columns contain different time series for
        the same id.

        :type df: pd.DataFrame
        :type column_id: str
        :type value_columns: list[str]
        """

        if column_id is None:
            raise ValueError("You have to set the column_id which contains the ids of the different time series")

        if column_id not in df.columns:
            raise AttributeError("The given column for the id is not present in the data.")

        if df[column_id].isnull().any():
            raise ValueError("You have NaN values in your id column.")

        if value_columns is None:
            value_columns = [col for col in df.columns if col not in [column_id, column_sort]]
        else:
            # Check value column
            for column_value in value_columns:
                if column_value not in df.columns:
                    raise ValueError(
                        "The given column for the value is not present in the data: {}.".format(column_value))

        if True in df[value_columns].isnull().any().values:
            raise ValueError("You have NaN values in your value columns.")

        self.value_columns = value_columns
        self.column_sort = column_sort

        if self.column_sort is not None:
            if df[column_sort].isnull().any():
                raise ValueError("You have NaN values in your sort column.")
            self.df_grouped = df.sort_values([self.column_sort]).groupby([column_id])
        else:
            self.df_grouped = df.groupby([column_id])

    def __iter__(self):
        return self.iter_slice(0)

    def iter_slice(self, offset, length=None):
        group_offset, column_offset = divmod(offset, len(self.value_columns))
        kinds = self.value_columns[column_offset:]
        i = 0

        for group_name, group in itertools.islice(self.df_grouped, group_offset, None):
            for kind in kinds:
                i += 1
                if length is not None and i > length:
                    return
                else:
                    yield group_name, kind, group[kind]
            kinds = self.value_columns

    def __len__(self):
        return self.df_grouped.ngroups * len(self.value_columns)

    def slice(self, offset, length=None):
        return _WideTsFrameAdapterSlice(self, offset, length)


class _WideTsFrameAdapterSlice(Iterable):
    def __init__(self, adapter, offset, length):
        """
        Wraps the iter_slice generator function so it can be pickled
        """
        self.adapter = adapter
        self.offset = offset
        self.length = length

    def __iter__(self):
        return self.adapter.iter_slice(self.offset, self.length)

    def __len__(self):
        return self.length


class LongTsFrameAdapter(TsData):

    def __init__(self, df, column_id, column_kind, column_value, column_sort=None):
        """
        Adapter for Pandas DataFrames in long format, where different time series for the same id are
        labeled by column `column_kind`.

        :type df: pd.DataFrame
        :type column_id: str
        :type column_kind: str
        :type column_value: str
        :type column_sort: str|None
        """

        if column_id is None:
            raise ValueError("You have to set the column_id which contains the ids of the different time series")

        if column_id not in df.columns:
            raise AttributeError("The given column for the id is not present in the data.")

        if df[column_id].isnull().any():
            raise ValueError("You have NaN values in your id column.")

        if column_kind is None:
            raise ValueError("You have to set the column_kind which contains the kinds of the different time series")

        if column_kind not in df.columns:
            raise AttributeError("The given column for the kind is not present in the data.")

        if df[column_kind].isnull().any():
            raise ValueError("You have NaN values in your kind column.")

        if column_value is None:
            raise ValueError("You have to set the column_value which contains the values of the different time series")

        # Check value column
        if column_value not in df.columns:
            raise ValueError("The given column for the value is not present in the data.")

        if df[column_value].isnull().any():
            raise ValueError("You have NaN values in your value column.")

        # Check sort column
        if column_sort is not None:
            if df[column_sort].isnull().any():
                raise ValueError("You have NaN values in your sort column.")

        self.column_value = column_value
        self.column_sort = column_sort
        self.df_grouped = df.groupby([column_id, column_kind])

    def __iter__(self):
        return self.iter_slice(0)

    def iter_slice(self, offset, length=None):
        length_or_none = None if length is None else offset + length

        for group_key, group in itertools.islice(self.df_grouped, offset, length_or_none):

            if self.column_sort is not None:
                group = group.sort_values([self.column_sort])

            yield group_key + (group[self.column_value],)

    def __len__(self):
        return len(self.df_grouped)

    def slice(self, offset, length=None):
        return _LongTsFrameAdapterSlice(self, offset, length)


class _LongTsFrameAdapterSlice(Iterable):
    def __init__(self, adapter, offset, length):
        """
        Wraps the iter_slice generator function so it can be pickled
        """
        self.adapter = adapter
        self.offset = offset
        self.length = length

    def __iter__(self):
        return self.adapter.iter_slice(self.offset, self.length)

    def __len__(self):
        return self.length


class TsDictAdapter(TsData):
    def __init__(self, ts_dict, column_id, column_value, column_sort=None, offset=0, length=None):
        """
        Adapter for a dict of Pandas DataFrames, which maps different time series kinds to Pandas DataFrames.

        :type ts_dict: dict
        :type column_id: str
        :type column_value: str
        :type column_sort: str|None
        :type offset: int
        :type length: int|None
        """

        if column_id is None:
            raise ValueError("You have to set the column_id which contains the ids of the different time series")

        if column_value is None:
            raise ValueError("You have to set the column_value which contains the values of the different time series")

        for key, df in ts_dict.items():
            if column_id not in df.columns:
                raise AttributeError("The given column for the id is not present in the data for key {}.".format(key))

            if df[column_id].isnull().any():
                raise ValueError("You have NaN values in your id column in data for key {}.".format(key))

            if column_id not in df.columns:
                raise AttributeError("The given column for the id is not present in data for key {}.".format(key))

            if df[column_id].isnull().any():
                raise ValueError("You have NaN values in your id column in data for key {}.".format(key))

            if column_value not in df.columns:
                raise ValueError("The given column for the value is not present in data for key {}.".format(key))

            if df[column_value].isnull().any():
                raise ValueError("You have NaN values in your value column in data for key {}.".format(key))

        self.ts_dict = ts_dict
        self.column_id = column_id
        self.column_value = column_value
        self.column_sort = column_sort
        self.offset = offset
        self.length = length

        self.grouped_dict = {key: df.groupby(column_id) for key, df in ts_dict.items()}

    def __iter__(self):
        for kind, grouped_df in self.grouped_dict.items():
            for ts_id, group in grouped_df:
                if self.column_sort is not None:
                    group = group.sort_values([self.column_sort])

                yield ts_id, kind, group[self.column_value]

    def __len__(self):
        return sum(grouped_df.ngroups for grouped_df in self.grouped_dict.values())


def to_tsdata(df, column_id=None, column_kind=None, column_value=None, column_sort=None):
    """Converts df in into an iterable of individual time series.

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
        if column_value is not None:
            # Check kind column
            if column_kind is not None:
                return LongTsFrameAdapter(df, column_id, column_kind, column_value, column_sort)
            else:
                return WideTsFrameAdapter(df, column_id, column_sort, [column_value])
        else:
            return WideTsFrameAdapter(df, column_id, column_sort)

    elif isinstance(df, dict):
        return TsDictAdapter(df, column_id, column_value, column_sort)

    else:
        raise ValueError("df must be a DataFrame or a dict of DataFrames. "
                         "See https://tsfresh.readthedocs.io/en/latest/text/data_formats.html")
