import itertools
import pandas as pd
from dask import dataframe as dd
from pandas.api.types import is_numeric_dtype
from typing import List


def engineer_input_timeseries(
    timeseries: pd.DataFrame,
    column_id: str = None,
    column_sort: str = None,
    compute_differences_within_series: bool = True,
    compute_differences_between_series: bool = False,
) -> pd.DataFrame:
    """
    time series differencing and phase difference operations to add new engineered time series to the input time series
    NOTE: The naming convention specified in P4P has been changed.

    params:
         ts (pd.DataFrame): time series input with n ts_kinds (n columns)
         compute_timeseries_differences (bool):
         compute_differences_between_series (bool):
         column_id (str):
         column_sort (str):
    """

    def series_differencing(ts: pd.DataFrame, ts_kinds: List[str]) -> pd.DataFrame:
        for ts_kind in ts_kinds:
            ts["dt_" + ts_kind] = ts[ts_kind].diff()
            ts.loc[
                0, ["dt_" + ts_kind]
            ] = 0  # adjust for the NaN value for temporal derivatives at first index...
        return ts

    def diff_between_series(ts: pd.DataFrame, ts_kinds: str) -> pd.DataFrame:
        assert (
            len(ts_kinds) > 1
        ), "Can only difference `ts` if there is more than one series"

        combs = itertools.combinations(ts_kinds, r=2)
        for first_ts_kind, second_ts_kind in combs:
            ts["D_" + first_ts_kind + second_ts_kind] = (
                ts[first_ts_kind] - ts[second_ts_kind]
            )
        return ts

    assert isinstance(timeseries, pd.DataFrame), "`ts` expected to be a pd.DataFrame"

    ts = timeseries.copy()

    ts_meta = ts[[column for column in [column_id, column_sort] if column is not None]]
    ts = ts.drop(
        [column for column in [column_id, column_sort] if column is not None], axis=1
    )

    assert all(
        is_numeric_dtype(ts[col]) for col in ts.columns.tolist()
    ), "All columns except `column_id` and `column_sort` in `ts` must be float or int"

    ts_kinds = ts.columns
    if compute_differences_within_series:
        ts = series_differencing(ts, ts_kinds)
    if compute_differences_between_series:
        ts = diff_between_series(ts, ts_kinds)

    return ts.join(ts_meta)
