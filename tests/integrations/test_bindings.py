from unittest import TestCase

import pandas as pd
from dask import dataframe as dd

from tsfresh.convenience.bindings import dask_feature_extraction_on_chunk
from tsfresh.feature_extraction.settings import MinimalFCParameters


class DaskBindingsTestCase(TestCase):
    def test_feature_extraction(self):
        df = pd.DataFrame(
            {
                "my_id": [1, 1, 1, 2, 2, 2],
                "my_kind": ["a"] * 6,
                "my_value": [1, 2, 3, 4, 5, 6],
            }
        )

        df = dd.from_pandas(df, chunksize=3)

        df_grouped = df.groupby(["my_id", "my_kind"])

        features = dask_feature_extraction_on_chunk(
            df_grouped,
            column_id="my_id",
            column_kind="my_kind",
            column_value="my_value",
            column_sort=None,
            default_fc_parameters=MinimalFCParameters(),
        )

        features = features.compute()

        self.assertEqual(list(sorted(features.columns)), ["my_id", "value", "variable"])
        self.assertEqual(len(features), 2 * len(MinimalFCParameters()))
