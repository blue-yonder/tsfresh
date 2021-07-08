from functools import partial

import pandas as pd

from tsfresh.feature_extraction.extraction import _do_extraction_on_chunk
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters


def _feature_extraction_on_chunk_helper(
    df,
    column_id,
    column_kind,
    column_sort,
    column_value,
    default_fc_parameters,
    kind_to_fc_parameters,
):
    """
    Helper function wrapped around _do_extraction_on_chunk to use the correct format
    of the "chunk" and output a pandas dataframe.
    Is used e.g. in the convenience functions for dask and spark.

    For the definitions of the parameters, please see these convenience functions.
    """
    if default_fc_parameters is None and kind_to_fc_parameters is None:
        default_fc_parameters = ComprehensiveFCParameters()
    elif default_fc_parameters is None and kind_to_fc_parameters is not None:
        default_fc_parameters = {}

    if column_sort is not None:
        df = df.sort_values(column_sort)

    chunk = df[column_id].iloc[0], df[column_kind].iloc[0], df[column_value]
    features = _do_extraction_on_chunk(
        chunk,
        default_fc_parameters=default_fc_parameters,
        kind_to_fc_parameters=kind_to_fc_parameters,
    )
    features = pd.DataFrame(features, columns=[column_id, "variable", "value"])
    features["value"] = features["value"].astype("double")

    return features[[column_id, "variable", "value"]]


def dask_feature_extraction_on_chunk(
    df,
    column_id,
    column_kind,
    column_value,
    column_sort=None,
    default_fc_parameters=None,
    kind_to_fc_parameters=None,
):
    """
    Extract features on a grouped dask dataframe given the column names and the extraction settings.
    This wrapper function should only be used if you have a dask dataframe as input.
    All format handling (input and output) needs to be done before or after that.

    Examples
    ========

    For example if you want to extract features on the robot example dataframe (stored as csv):

    Import statements:

    >>>  from dask import dataframe as dd
    >>>  from tsfresh.convenience.bindings import dask_feature_extraction_on_chunk
    >>>  from tsfresh.feature_extraction.settings import MinimalFCParameters

    Read in the data

    >>>  df = dd.read_csv("robot.csv")

    Prepare the data into correct format.
    The format needs to be a grouped dataframe (grouped by time series id and feature kind),
    where each group chunk consists of a dataframe with exactly 4 columns: ``column_id``,
    ``column_kind``, ``column_sort`` and ``column_value``.
    You can find the description of the columns in :ref:`data-formats-label`.
    Please note: for this function to work you need to have all columns present!
    If necessary create the columns and fill them with dummy values.

    >>>  df = df.melt(id_vars=["id", "time"],
    ...               value_vars=["F_x", "F_y", "F_z", "T_x", "T_y", "T_z"],
    ...               var_name="kind", value_name="value")
    >>>  df_grouped = df.groupby(["id", "kind"])

    Call the feature extraction

    >>>  features = dask_feature_extraction_on_chunk(df_grouped, column_id="id", column_kind="kind",
    ...                                              column_sort="time", column_value="value",
    ...                                              default_fc_parameters=MinimalFCParameters())

    Write out the data in a tabular format

    >>>  features = features.categorize(columns=["variable"])
    >>>  features = features.reset_index(drop=True) \\
    ...                 .pivot_table(index="id", columns="variable", values="value", aggfunc="mean")
    >>>  features.to_csv("output")


    :param df: A dask dataframe grouped by id and kind.
    :type df: dask.dataframe.groupby.DataFrameGroupBy

    :param default_fc_parameters: mapping from feature calculator names to parameters. Only those names
           which are keys in this dict will be calculated. See the class:`ComprehensiveFCParameters` for
           more information.
    :type default_fc_parameters: dict

    :param kind_to_fc_parameters: mapping from kind names to objects of the same type as the ones for
            default_fc_parameters. If you put a kind as a key here, the fc_parameters
            object (which is the value), will be used instead of the default_fc_parameters. This means
            that kinds, for which kind_of_fc_parameters doe not have any entries, will be ignored by
            the feature selection.
    :type kind_to_fc_parameters: dict

    :param column_id: The name of the id column to group by.
    :type column_id: str

    :param column_sort: The name of the sort column.
    :type column_sort: str or None

    :param column_kind: The name of the column keeping record on the kind of the value.
    :type column_kind: str

    :param column_value: The name for the column keeping the value itself.
    :type column_value: str

    :return: A dask dataframe with the columns ``column_id``, "variable" and "value". The index is taken
            from the grouped dataframe.
    :rtype: dask.dataframe.DataFrame (id int64, variable object, value float64)

    """
    feature_extraction = partial(
        _feature_extraction_on_chunk_helper,
        column_id=column_id,
        column_kind=column_kind,
        column_sort=column_sort,
        column_value=column_value,
        default_fc_parameters=default_fc_parameters,
        kind_to_fc_parameters=kind_to_fc_parameters,
    )
    return df.apply(
        feature_extraction,
        meta=[(column_id, "int64"), ("variable", "object"), ("value", "float64")],
    )


def spark_feature_extraction_on_chunk(
    df,
    column_id,
    column_kind,
    column_value,
    column_sort=None,
    default_fc_parameters=None,
    kind_to_fc_parameters=None,
):
    """
    Extract features on a grouped spark dataframe given the column names and the extraction settings.
    This wrapper function should only be used if you have a spark dataframe as input.
    All format handling (input and output) needs to be done before or after that.

    Examples
    ========

    For example if you want to extract features on the robot example dataframe (stored as csv):

    Import statements:

    >>>  from tsfresh.convenience.bindings import spark_feature_extraction_on_chunk
    >>>  from tsfresh.feature_extraction.settings import MinimalFCParameters

    Read in the data

    >>>  df = spark.read(...)

    Prepare the data into correct format.
    The format needs to be a grouped dataframe (grouped by time series id and feature kind),
    where each group chunk consists of a dataframe with exactly 4 columns: ``column_id``,
    ``column_kind``, ``column_sort`` and ``column_value``.
    You can find the description of the columns in :ref:`data-formats-label`.
    Please note: for this function to work you need to have all columns present!
    If necessary create the columns and fill them with dummy values.

    >>>  df = ...
    >>>  df_grouped = df.groupby(["id", "kind"])

    Call the feature extraction

    >>>  features = spark_feature_extraction_on_chunk(df_grouped, column_id="id", column_kind="kind",
    ...                                               column_sort="time", column_value="value",
    ...                                               default_fc_parameters=MinimalFCParameters())

    Write out the data in a tabular format

    >>>  features = features.groupby("id").pivot("variable").sum("value")
    >>>  features.write.csv("output")


    :param df: A spark dataframe grouped by id and kind.
    :type df: pyspark.sql.group.GroupedData

    :param default_fc_parameters: mapping from feature calculator names to parameters. Only those names
           which are keys in this dict will be calculated. See the class:`ComprehensiveFCParameters` for
           more information.
    :type default_fc_parameters: dict

    :param kind_to_fc_parameters: mapping from kind names to objects of the same type as the ones for
            default_fc_parameters. If you put a kind as a key here, the fc_parameters
            object (which is the value), will be used instead of the default_fc_parameters.
            This means that kinds, for which kind_of_fc_parameters doe not have any entries,
            will be ignored by the feature selection.
    :type kind_to_fc_parameters: dict

    :param column_id: The name of the id column to group by.
    :type column_id: str

    :param column_sort: The name of the sort column.
    :type column_sort: str or None

    :param column_kind: The name of the column keeping record on the kind of the value.
    :type column_kind: str

    :param column_value: The name for the column keeping the value itself.
    :type column_value: str

    :return: A dask dataframe with the columns ``column_id``, "variable" and "value".
    :rtype: pyspark.sql.DataFrame[id: bigint, variable: string, value: double]

    """
    from pyspark.sql.functions import PandasUDFType, pandas_udf

    feature_extraction = partial(
        _feature_extraction_on_chunk_helper,
        column_id=column_id,
        column_kind=column_kind,
        column_sort=column_sort,
        column_value=column_value,
        default_fc_parameters=default_fc_parameters,
        kind_to_fc_parameters=kind_to_fc_parameters,
    )

    type_string = "{column_id} long, variable string, value double".format(
        column_id=column_id
    )
    feature_extraction_udf = pandas_udf(type_string, PandasUDFType.GROUPED_MAP)(
        feature_extraction
    )

    return df.apply(feature_extraction_udf)
