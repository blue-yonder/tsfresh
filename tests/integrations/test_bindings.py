from unittest import TestCase

from tsfresh.convenience.bindings import dask_feature_extraction_on_chunk
from tsfresh.feature_extraction.settings import MinimalFCParameters

from dask import dataframe as dd
import pandas as pd


class DaskBindingsTestCase(TestCase):
    def test_feature_extraction(self):
        df = pd.DataFrame({"my_id": [1, 1, 1, 2, 2, 2], "my_kind": ["a"]*6,
                           "my_value": [1, 2, 3, 4, 5, 6]})

        df = dd.from_pandas(df, chunksize=3)

        df_grouped = df.groupby(["my_id", "my_kind"])

        features = dask_feature_extraction_on_chunk(df_grouped, column_id="my_id",
                                                    column_kind="my_kind",
                                                    column_value="my_value",
                                                    column_sort=None,
                                                    default_fc_parameters=MinimalFCParameters())

        features = features.categorize(columns=["variable"])
        features = features.reset_index(drop=True)

        feature_table = features.pivot_table(index="my_id", columns="variable", values="value", aggfunc="sum")

        feature_table = feature_table.compute()

        self.assertEqual(len(feature_table.columns), len(MinimalFCParameters()))
        self.assertEqual(len(feature_table), 2)
