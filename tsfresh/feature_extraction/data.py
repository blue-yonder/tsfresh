import itertools
from collections import namedtuple, defaultdict
from typing import Generator, Iterable, Sized

import pandas as pd
try:
    from dask import dataframe as dd
except ImportError:
    dd = None


def _binding_helper(f, kwargs, column_sort, column_id, column_kind, column_value):
    def wrapped_feature_extraction(x):
        if column_sort is not None:
            x = x.sort_values(column_sort)

        chunk = Timeseries(x[column_id].iloc[0], x[column_kind].iloc[0], x[column_value])
        result = f(chunk, **kwargs)

        result = pd.DataFrame(result, columns=[column_id, "variable", "value"])
        result["value"] = result["value"].astype("double")

        return result[[column_id, "variable", "value"]]

    return wrapped_feature_extraction


class Timeseries(namedtuple('Timeseries', ['id', 'kind', 'data'])):
    """
    Timeseries tuple used for feature extraction.

    Make sure `kind` is of type `str` to allow inference
    of feature settings in `feature_extraction.settings.from_columns`.
    """


class TsData:
    """
    TsData provides access to time series data for internal usage.

    Distributors will use this data class to apply functions on the data.
    All derived classes must either implement the `apply` method,
    which is used to apply the given function directly on the data
    or the __iter__ method, which can be used to get an iterator of
    Timeseries instances (which distributors can use to apply the function on).
    Other methods can be overwritten if a more efficient solution exists for the underlying data store.
    """
    pass


class PartitionedTsData(Iterable[Timeseries], Sized):
    """
    Special class of TsData, which can be partitioned.
    Derived classes should implement __iter__ and __len__.
    """
    def __init__(self, df, column_id):
        self.df_id_type = df[column_id].dtype

    def partition(self, chunk_size):
        """
        Split the data into iterable chunks of given `chunk_size`.

        :param chunk_size: the size of the chunks
        :type chunk_size: int

        :return: chunks with at most `chunk_size` items
        :rtype: Generator[Iterable[Timeseries]]
        """
        iterable = iter(self)
        while True:
            next_chunk = list(itertools.islice(iterable, chunk_size))
            if not next_chunk:
                return

            yield next_chunk

    def pivot(self, results):
        """
        Helper function to turn an iterable of tuples with three entries into a dataframe.

        The input ``list_of_tuples`` needs to be an iterable with tuples containing three
        entries: (a, b, c).
        Out of this, a pandas dataframe will be created with all a's as index,
        all b's as columns and all c's as values.

        It basically does a pd.pivot(first entry, second entry, third entry),
        but optimized for non-pandas input (= python list of tuples).

        This function is called in the end of the extract_features call.
        """
        return_df_dict = defaultdict(dict)
        for chunk_id, variable, value in results:
            # we turn it into a nested mapping `column -> index -> value`
            return_df_dict[variable][chunk_id] = value

        # the mapping column -> {index -> value}
        # is now a dict of dicts. The pandas dataframe
        # constructor will peel this off:
        # first, the keys of the outer dict (the column)
        # will turn into a column header and the rest into a column
        # the rest is {index -> value} which will be turned into a
        # column with index.
        # All index will be aligned.
        return_df = pd.DataFrame(return_df_dict, dtype=float)

        # copy the type of the index
        return_df.index = return_df.index.astype(self.df_id_type)

        # Sort by index to be backward compatible
        return_df = return_df.sort_index()

        return return_df


class SliceableTsData(PartitionedTsData):
    """
    SliceableTsData uses a slice strategy to implement data partitioning.
    Implementations of `TsData` may extend this class if they can iterate their elements in deterministic order and
    if an efficient strategy exists to slice the data.

    Because `__iter__` defaults to `slice(0)`, implementations must only implement `slice`.
    """

    def slice(self, offset, length=None):
        """
        Get a subset of the timeseries elements

        :param offset: the offset in the sequence of elements
        :type offset: int

        :param length: if not `None`, the maximum number of elements the slice will yield
        :type length: int

        :return: a slice of the data
        :rtype: Iterable[Timeseries]
        """
        raise NotImplementedError

    def __iter__(self):
        return self.slice(0)

    def partition(self, chunk_size):
        div, mod = divmod(len(self), chunk_size)
        chunk_info = [(i * chunk_size, chunk_size) for i in range(div)]
        if mod > 0:
            chunk_info += [(div * chunk_size, mod)]

        for offset, length in chunk_info:
            yield _Slice(self, offset, length)


class _Slice(Iterable[Timeseries]):
    def __init__(self, data, offset, length):
        """
        Wraps the `slice` generator function as an iterable object which can be pickled and passed to the distributor
        backend.

        :type data: SliceableTsData
        :type offset: int
        :type length: int
        """
        self.data = data
        self.offset = offset
        self.length = length

    def __iter__(self):
        return self.data.slice(self.offset, self.length)

    def __len__(self):
        return self.length


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

        if value_columns is None:
            value_columns = _get_value_columns(df, column_id, column_sort)

        _check_nan(df, *value_columns)
        _check_colname(*value_columns)

        self.value_columns = value_columns

        if column_sort is not None:
            _check_nan(df, column_sort)
            self.df_grouped = df.sort_values([column_sort]).groupby([column_id])
        else:
            self.df_grouped = df.groupby([column_id])

        super().__init__(df, column_id)

    def __len__(self):
        return self.df_grouped.ngroups * len(self.value_columns)

    def slice(self, offset, length=None):
        if 0 < offset >= len(self):
            raise ValueError("offset out of range: {}".format(offset))

        group_offset, column_offset = divmod(offset, len(self.value_columns))
        kinds = self.value_columns[column_offset:]
        i = 0

        for group_name, group in itertools.islice(self.df_grouped, group_offset, None):
            for kind in kinds:
                i += 1
                if length is not None and i > length:
                    return
                yield Timeseries(group_name, kind, group[kind])
            kinds = self.value_columns


class LongTsFrameAdapter(PartitionedTsData):
    def __init__(self, df, column_id, column_kind, column_value, column_sort=None):
        """
        Adapter for Pandas DataFrames in long format, where different time series for the same id are
        labeled by column `column_kind`.

        :param df: the data frame
        :type df: pd.DataFrame

        :param column_id: the name of the column containing time series group ids
        :type column_id: str

        :param column_kind: the name of the column containing time series kinds for each id
        :type column_kind: str

        :param column_value: the name of the column containing time series values
        :type column_value: str

        :param column_sort: the name of the column to sort on
        :type column_sort: str|None
        """
        _check_nan(df, column_id, column_kind, column_value)
        if column_id is None:
            raise ValueError("A value for column_id needs to be supplied")
        if column_kind is None:
            raise ValueError("A value for column_kind needs to be supplied")

        if column_value is None:
            possible_value_columns = _get_value_columns(df, column_id, column_sort, column_kind)
            if len(possible_value_columns) != 1:
                raise ValueError("Could not guess the value column! Please hand it to the function as an argument.")
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

        super().__init__(df, column_id)

    def __len__(self):
        return len(self.df_grouped)

    def __iter__(self):
        for group_key, group in self.df_grouped:
            yield Timeseries(group_key[0], str(group_key[1]), group[self.column_value])


class TsDictAdapter(PartitionedTsData):
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

        super().__init__(df, column_id)

    def __iter__(self):
        for kind, grouped_df in self.grouped_dict.items():
            for ts_id, group in grouped_df:
                yield Timeseries(ts_id, str(kind), group[self.column_value])

    def __len__(self):
        return sum(grouped_df.ngroups for grouped_df in self.grouped_dict.values())


class DaskTsAdapter(TsData):
    def __init__(self, df, column_id, column_kind=None, column_value=None, column_sort=None):
        assert column_id is not None, "column_id must be set"

        # Get all columns, which are not id, kind or sort
        possible_value_columns = _get_value_columns(df, column_id, column_sort, column_kind)

        # The user has already a kind column. That means we just need to group by id (and additionally by id)
        if column_kind is not None:
            self.df = df.groupby([column_id, column_kind])

            # We assume the last remaining column is the value - but there needs to be one!
            if column_value is None:
                if len(possible_value_columns) != 1:
                    raise ValueError("Could not guess the value column! Please hand it to the function as an argument.")
                column_value = possible_value_columns[0]
        else:
            # Ok, the user has no kind, so it is in Wide format.
            # That means we have do melt before we can group.
            # TODO: here is some room for optimization!

            # We first choose a name for our future kind column
            column_kind = "kind"

            # if the user has specified a value column, use it
            # if not, just go with every remaining columns
            if column_value is not None:
                value_vars = [column_value]
            else:
                value_vars = possible_value_columns
                column_value = "value"

            id_vars = [column_id, column_sort] if column_sort else [column_id]

            # Now melt and group
            df_melted = df.melt(id_vars=id_vars, value_vars=value_vars,
                                var_name=column_kind, value_name=column_value)

            self.df = df_melted.groupby([column_id, column_kind])

        self.column_id = column_id
        self.column_kind = column_kind
        self.column_value = column_value
        self.column_sort = column_sort

    def apply(self, f, meta, **kwargs):
        """
        Apply the wrapped feature extraction function "f"
        onto the data.
        Before that, turn the data into the correct form of Timeseries instances
        usable the the feature extraction.
        After the call, turn it back into pandas dataframes
        for further processing.
        """
        bound_function = _binding_helper(f, kwargs, self.column_sort, self.column_id, self.column_kind, self.column_value)
        return self.df.apply(bound_function, meta=meta)

    def pivot(self, results):
        """
        The extract features function for dask returns a
        dataframe of [id, variable, value].
        Turn this into a pivoted dataframe, where only the variables are the columns
        and the ids are the rows.

        Attention: this is highly non-optimized!
        """
        results = results.reset_index(drop=True).persist()
        results = results.categorize(columns=["variable"])
        feature_table = results.pivot_table(index=self.column_id, columns="variable",
                                            values="value", aggfunc="sum")

        return feature_table


def to_tsdata(df, column_id, column_kind=None, column_value=None, column_sort=None):
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
    :type column_id: str

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

    elif dd and isinstance(df, dd.DataFrame):
        return DaskTsAdapter(df, column_id, column_kind, column_value, column_sort)

    else:
        raise ValueError("df must be a DataFrame or a dict of DataFrames. "
                         "See https://tsfresh.readthedocs.io/en/latest/text/data_formats.html")
