# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from unittest import TestCase

import numpy as np
import pandas as pd
import pandas.testing as pdt

from tests.fixtures import DataTestCase
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute


class RelevantFeatureExtractionDataTestCase(DataTestCase):
    """
    Test case for the relevant_feature_extraction function
    """

    def test_functional_equality(self):
        """
        `extract_relevant_features` should be equivalent to running first `extract_features` with impute and
        `select_features` afterwards.
        Meaning it should produce the same relevant features and the values of these features should be identical.
        :return:
        """
        df, y = self.create_test_data_sample_with_target()

        relevant_features = extract_relevant_features(
            df,
            y,
            column_id="id",
            column_value="val",
            column_kind="kind",
            column_sort="sort",
        )

        extracted_features = extract_features(
            df,
            column_id="id",
            column_value="val",
            column_kind="kind",
            column_sort="sort",
            impute_function=impute,
        )
        selected_features = select_features(extracted_features, y)

        self.assertEqual(
            set(relevant_features.columns),
            set(selected_features.columns),
            "Should select the same columns:\n\t{}\n\nvs.\n\n\t{}".format(
                relevant_features.columns, selected_features.columns
            ),
        )

        relevant_columns = relevant_features.columns
        relevant_index = relevant_features.index
        self.assertTrue(
            relevant_features.equals(
                selected_features.loc[relevant_index][relevant_columns]
            ),
            "Should calculate the same feature values",
        )


class RelevantFeatureExtractionTestCase(TestCase):
    def setUp(self):
        np.random.seed(42)
        y = pd.Series(np.random.binomial(1, 0.5, 20), index=range(20))
        df = pd.DataFrame(index=range(100))

        df["a"] = np.random.normal(0, 1, 100)
        df["b"] = np.random.normal(0, 1, 100)
        df["id"] = np.repeat(range(20), 5)

        X = pd.DataFrame(index=range(20))
        X["f1"] = np.random.normal(0, 1, 20)
        X["f2"] = np.random.normal(0, 1, 20)

        self.df = df
        self.X = X
        self.y = y

    def test_extracted_features_contain_X_features(self):
        X = extract_relevant_features(self.df, self.y, self.X, column_id="id")
        self.assertIn("f1", X.columns)
        self.assertIn("f2", X.columns)
        pdt.assert_series_equal(self.X["f1"], X["f1"])
        pdt.assert_series_equal(self.X["f2"], X["f2"])
        pdt.assert_index_equal(self.X["f1"].index, X["f1"].index)
        pdt.assert_index_equal(self.X["f2"].index, X["f2"].index)

    def test_extraction_null_as_column_name(self):

        df1 = pd.DataFrame(
            data={
                0: range(10),
                1: np.repeat([0, 1], 5),
                2: np.repeat([0, 1, 2, 3, 4], 2),
            }
        )
        X1 = extract_features(df1, column_id=1, column_sort=2)
        self.assertEqual(len(X1), 2)

        df2 = pd.DataFrame(
            data={
                1: range(10),
                0: np.repeat([0, 1], 5),
                2: np.repeat([0, 1, 2, 3, 4], 2),
            }
        )
        X2 = extract_features(df2, column_id=0, column_sort=2)
        self.assertEqual(len(X2), 2)

        df3 = pd.DataFrame(
            data={
                0: range(10),
                2: np.repeat([0, 1], 5),
                1: np.repeat([0, 1, 2, 3, 4], 2),
            }
        )
        X3 = extract_features(df3, column_id=2, column_sort=1)
        self.assertEqual(len(X3), 2)

    def test_raises_mismatch_index_df_and_y_df_more(self):
        y = pd.Series(range(3), index=[1, 2, 3])
        df_dict = {
            "a": pd.DataFrame({"val": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]}),
            "b": pd.DataFrame({"val": [5, 6, 7, 8, 12, 13], "id": [4, 4, 3, 3, 2, 2]}),
        }
        self.assertRaises(
            ValueError,
            extract_relevant_features,
            df_dict,
            y,
            None,
            None,
            None,
            "id",
            None,
            "val",
        )

    def test_raises_mismatch_index_df_and_y_y_more(self):
        y = pd.Series(range(4), index=[1, 2, 3, 4])
        df = pd.DataFrame({"val": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]})
        self.assertRaises(
            ValueError,
            extract_relevant_features,
            df,
            y,
            None,
            None,
            None,
            "id",
            None,
            "val",
        )

    def test_raises_y_not_series(self):
        y = np.arange(10)
        df_dict = {
            "a": pd.DataFrame({"val": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]}),
            "b": pd.DataFrame({"val": [5, 6, 7, 8, 12, 13], "id": [4, 4, 3, 3, 2, 2]}),
        }
        self.assertRaises(
            AssertionError,
            extract_relevant_features,
            df_dict,
            y,
            None,
            None,
            None,
            "id",
            None,
            "val",
        )

    def test_raises_y_not_more_than_one_label(self):
        y = pd.Series(1, index=[1, 2, 3])
        df_dict = {
            "a": pd.DataFrame({"val": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]}),
            "b": pd.DataFrame({"val": [5, 6, 7, 8, 12, 13], "id": [4, 4, 3, 3, 2, 2]}),
        }
        self.assertRaises(
            AssertionError,
            extract_relevant_features,
            df_dict,
            y,
            None,
            None,
            None,
            "id",
            None,
            "val",
        )
