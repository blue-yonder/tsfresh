# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from unittest import TestCase

import dask.dataframe as dd
import pandas as pd

from tsfresh import extract_features, extract_relevant_features
from tsfresh.examples.driftbif_simulation import load_driftbif
from tsfresh.feature_extraction import MinimalFCParameters


class FeatureExtractionTestCase(TestCase):
    def setUp(self):
        df, y = load_driftbif(100, 10, classification=True, seed=42)

        df["my_id"] = df["id"].astype("str")
        del df["id"]

        self.df = df

    def test_pandas(self):
        df = self.df

        # Test shape and a single entry (to see if it works at all)
        X = extract_features(
            df,
            column_id="my_id",
            column_sort="time",
            column_kind="dimension",
            column_value="value",
            default_fc_parameters=MinimalFCParameters(),
        )
        self.assertIn("1__mean", X.columns)
        self.assertAlmostEqual(X.loc["5", "1__mean"], 5.516e-05, 4)
        self.assertIn("11", X.index)
        self.assertEqual(X.shape, (100, 20))

        X = extract_features(
            df,
            column_id="my_id",
            column_sort="time",
            column_kind="dimension",
            default_fc_parameters=MinimalFCParameters(),
        )
        self.assertIn("1__mean", X.columns)
        self.assertAlmostEqual(X.loc["5", "1__mean"], 5.516e-05, 4)
        self.assertIn("11", X.index)
        self.assertEqual(X.shape, (100, 20))

        X = extract_features(
            df.drop(columns=["dimension"]),
            column_id="my_id",
            column_sort="time",
            default_fc_parameters=MinimalFCParameters(),
        )
        self.assertIn("value__mean", X.columns)
        self.assertAlmostEqual(X.loc["5", "value__mean"], 5.516e-05, 4)
        self.assertIn("11", X.index)
        self.assertEqual(X.shape, (100, 10))

        X = extract_features(
            df.drop(columns=["dimension", "time"]),
            column_id="my_id",
            default_fc_parameters=MinimalFCParameters(),
        )
        self.assertIn("value__mean", X.columns)
        self.assertAlmostEqual(X.loc["5", "value__mean"], 5.516e-05, 4)
        self.assertIn("11", X.index)
        self.assertEqual(X.shape, (100, 10))

    def test_pandas_no_pivot(self):
        df = self.df

        X = extract_features(
            df,
            column_id="my_id",
            column_sort="time",
            column_kind="dimension",
            column_value="value",
            pivot=False,
            default_fc_parameters=MinimalFCParameters(),
        )
        X = pd.DataFrame(X, columns=["my_id", "variable", "value"])
        self.assertIn("1__mean", X["variable"].values)
        self.assertAlmostEqual(
            X[(X["my_id"] == "5") & (X["variable"] == "1__mean")]["value"].iloc[0],
            5.516e-05,
            4,
        )
        self.assertEqual(X.shape, (100 * 20, 3))

        X = extract_features(
            df,
            column_id="my_id",
            column_sort="time",
            column_kind="dimension",
            pivot=False,
            default_fc_parameters=MinimalFCParameters(),
        )
        X = pd.DataFrame(X, columns=["my_id", "variable", "value"])
        self.assertIn("1__mean", X["variable"].values)
        self.assertAlmostEqual(
            X[(X["my_id"] == "5") & (X["variable"] == "1__mean")]["value"].iloc[0],
            5.516e-05,
            4,
        )
        self.assertEqual(X.shape, (100 * 20, 3))

        X = extract_features(
            df.drop(columns=["dimension"]),
            column_id="my_id",
            column_sort="time",
            pivot=False,
            default_fc_parameters=MinimalFCParameters(),
        )
        X = pd.DataFrame(X, columns=["my_id", "variable", "value"])
        self.assertIn("value__mean", X["variable"].values)
        self.assertAlmostEqual(
            X[(X["my_id"] == "5") & (X["variable"] == "value__mean")]["value"].iloc[0],
            5.516e-05,
            4,
        )
        self.assertEqual(X.shape, (100 * 10, 3))

        X = extract_features(
            df.drop(columns=["dimension", "time"]),
            column_id="my_id",
            pivot=False,
            default_fc_parameters=MinimalFCParameters(),
        )
        X = pd.DataFrame(X, columns=["my_id", "variable", "value"])
        self.assertIn("value__mean", X["variable"].values)
        self.assertAlmostEqual(
            X[(X["my_id"] == "5") & (X["variable"] == "value__mean")]["value"].iloc[0],
            5.516e-05,
            4,
        )
        self.assertEqual(X.shape, (100 * 10, 3))

    def test_dask(self):
        df = dd.from_pandas(self.df, npartitions=1)

        X = extract_features(
            df,
            column_id="my_id",
            column_sort="time",
            column_kind="dimension",
            column_value="value",
            default_fc_parameters=MinimalFCParameters(),
        ).compute()
        self.assertIn("1__mean", X.columns)
        self.assertAlmostEqual(X.loc["5", "1__mean"], 5.516e-05, 4)
        self.assertIn("11", X.index)
        self.assertEqual(X.shape, (100, 20))

        X = extract_features(
            df,
            column_id="my_id",
            column_sort="time",
            column_kind="dimension",
            default_fc_parameters=MinimalFCParameters(),
        ).compute()
        self.assertIn("1__mean", X.columns)
        self.assertAlmostEqual(X.loc["5", "1__mean"], 5.516e-05, 4)
        self.assertIn("11", X.index)
        self.assertEqual(X.shape, (100, 20))

        X = extract_features(
            df.drop(columns=["dimension"]),
            column_id="my_id",
            column_sort="time",
            default_fc_parameters=MinimalFCParameters(),
        ).compute()
        self.assertIn("value__mean", X.columns)
        self.assertAlmostEqual(X.loc["5", "value__mean"], 5.516e-05, 4)
        self.assertIn("11", X.index)
        self.assertEqual(X.shape, (100, 10))

        X = extract_features(
            df.drop(columns=["dimension", "time"]),
            column_id="my_id",
            default_fc_parameters=MinimalFCParameters(),
        ).compute()
        self.assertIn("value__mean", X.columns)
        self.assertAlmostEqual(X.loc["5", "value__mean"], 5.516e-05, 4)
        self.assertIn("11", X.index)
        self.assertEqual(X.shape, (100, 10))

    def test_dask_no_pivot(self):
        df = dd.from_pandas(self.df, npartitions=1)

        X = extract_features(
            df,
            column_id="my_id",
            column_sort="time",
            column_kind="dimension",
            column_value="value",
            pivot=False,
            default_fc_parameters=MinimalFCParameters(),
        ).compute()
        self.assertIn("1__mean", X["variable"].values)
        self.assertAlmostEqual(
            X[(X["my_id"] == "5") & (X["variable"] == "1__mean")]["value"].iloc[0],
            5.516e-05,
            4,
        )
        self.assertEqual(X.shape, (100 * 20, 3))

        X = extract_features(
            df,
            column_id="my_id",
            column_sort="time",
            column_kind="dimension",
            pivot=False,
            default_fc_parameters=MinimalFCParameters(),
        ).compute()
        self.assertIn("1__mean", X["variable"].values)
        self.assertAlmostEqual(
            X[(X["my_id"] == "5") & (X["variable"] == "1__mean")]["value"].iloc[0],
            5.516e-05,
            4,
        )
        self.assertEqual(X.shape, (100 * 20, 3))

        X = extract_features(
            df.drop(columns=["dimension"]),
            column_id="my_id",
            column_sort="time",
            pivot=False,
            default_fc_parameters=MinimalFCParameters(),
        ).compute()
        self.assertIn("value__mean", X["variable"].values)
        self.assertAlmostEqual(
            X[(X["my_id"] == "5") & (X["variable"] == "value__mean")]["value"].iloc[0],
            5.516e-05,
            4,
        )
        self.assertEqual(X.shape, (100 * 10, 3))

        X = extract_features(
            df.drop(columns=["dimension", "time"]),
            column_id="my_id",
            pivot=False,
            default_fc_parameters=MinimalFCParameters(),
        ).compute()
        self.assertIn("value__mean", X["variable"].values)
        self.assertAlmostEqual(
            X[(X["my_id"] == "5") & (X["variable"] == "value__mean")]["value"].iloc[0],
            5.516e-05,
            4,
        )
        self.assertEqual(X.shape, (100 * 10, 3))
