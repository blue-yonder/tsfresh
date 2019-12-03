# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
import warnings
from unittest import TestCase

import pandas as pd
from tsfresh.utilities import dataframe_functions
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

from tsfresh.utilities.dataframe_functions import get_ids


class NormalizeTestCase(TestCase):
    def test_with_dictionaries_one_row(self):
        test_df = pd.DataFrame([{"value": 1, "id": "id_1"}])
        test_dict = {"a": test_df, "b": test_df}

        # A kind is not allowed with dicts
        self.assertRaises(ValueError, dataframe_functions._normalize_input_to_internal_representation, test_dict,
                          "id", None, "a kind", None)

        # The value must be present
        self.assertRaises(ValueError, dataframe_functions._normalize_input_to_internal_representation, test_dict,
                          "id", None, None, "something other")

    def test_with_dictionaries_two_rows(self):
        test_df = pd.DataFrame([{"value": 2, "sort": 2, "id": "id_1"},
                                {"value": 1, "sort": 1, "id": "id_1"}])
        test_dict = {"a": test_df, "b": test_df}

        # If there are more than one column, the algorithm can not choose the correct column
        self.assertRaises(ValueError, dataframe_functions._normalize_input_to_internal_representation, test_dict,
                          "id", None, None, None)

        # Sorting should work
        result_df, column_id, column_kind, column_value = \
            dataframe_functions._normalize_input_to_internal_representation(test_dict, "id", "sort", None, "value")
        self.assertEqual(column_value, "value")
        self.assertEqual(column_id, "id")

        # Assert sorted and without sort column
        self.assertEqual(result_df[result_df[column_kind] == "a"].iloc[0].to_dict(),
                         {"_variables": "a", "value": 1, "id": "id_1"})
        self.assertEqual(result_df[result_df[column_kind] == "a"].iloc[1].to_dict(),
                         {"_variables": "a", "value": 2, "id": "id_1"})

    def test_with_dictionaries_two_rows_sorted(self):
        test_df = pd.DataFrame([{"value": 2, "id": "id_1"},
                                {"value": 1, "id": "id_1"}])
        test_dict = {"a": test_df, "b": test_df}

        # Pass the id
        result_df, column_id, column_kind, column_value = \
            dataframe_functions._normalize_input_to_internal_representation(test_dict, "id", None, None, "value")
        self.assertEqual(column_value, "value")
        self.assertEqual(column_id, "id")

        self.assertEqual(result_df[result_df[column_kind] == "a"].iloc[0].to_dict(),
                         {"_variables": "a", "value": 2, "id": "id_1"})

    def test_with_df_1(self):
        # give everyting
        test_df = pd.DataFrame([{"id": 0, "kind": "a", "value": 3, "sort": 1}])
        result_df, column_id, column_kind, column_value = \
            dataframe_functions._normalize_input_to_internal_representation(test_df, "id", "sort", "kind", "value")

        self.assertEqual(column_id, "id")
        self.assertEqual(column_value, "value")
        self.assertEqual(column_kind, "kind")
        self.assertIn("a", set(result_df[column_kind]))
        self.assertCountEqual(list(result_df.columns), ["id", "value", "kind"])
        self.assertEqual(list(result_df[result_df[column_kind] == "a"]["value"]), [3])
        self.assertEqual(list(result_df[result_df[column_kind] == "a"]["id"]), [0])

    def test_with_df_2(self):
        # give no kind
        test_df = pd.DataFrame([{"id": 0, "value": 3, "sort": 1}])
        result_df, column_id, column_kind, column_value = \
            dataframe_functions._normalize_input_to_internal_representation(test_df, "id", "sort", None, "value")

        self.assertEqual(column_id, "id")
        self.assertEqual(column_value, "value")
        self.assertEqual(column_kind, "_variables")
        self.assertIn("value", set(result_df[column_kind]))
        self.assertCountEqual(list(result_df.columns), ["id", "value", "_variables"])
        self.assertEqual(list(result_df[result_df[column_kind] == "value"]["value"]), [3])
        self.assertEqual(list(result_df[result_df[column_kind] == "value"]["id"]), [0])

    def test_with_df_3(self):
        # Let the function find the values
        test_df = pd.DataFrame([{"id": 0, "a": 3, "b": 5, "sort": 1}])
        result_df, column_id, column_kind, column_value = \
            dataframe_functions._normalize_input_to_internal_representation(test_df, "id", "sort", None, None)

        self.assertEqual(column_id, "id")
        self.assertEqual(column_value, "_values")
        self.assertEqual(column_kind, "_variables")
        self.assertIn("a", set(result_df[column_kind]))
        self.assertIn("b", set(result_df[column_kind]))
        self.assertCountEqual(list(result_df.columns), ["_values", "_variables", "id"])
        self.assertEqual(list(result_df[result_df[column_kind] == "a"]["_values"]), [3])
        self.assertEqual(list(result_df[result_df[column_kind] == "a"]["id"]), [0])
        self.assertEqual(list(result_df[result_df[column_kind] == "b"]["_values"]), [5])
        self.assertEqual(list(result_df[result_df[column_kind] == "b"]["id"]), [0])

    def test_with_wrong_input(self):
        test_df = pd.DataFrame([{"id": 0, "kind": "a", "value": 3, "sort": np.NaN}])
        self.assertRaises(ValueError, dataframe_functions._normalize_input_to_internal_representation, test_df,
                          "id", "sort", "kind", "value")

        test_df = pd.DataFrame([{"id": 0, "kind": "a", "value": 3, "sort": 1}])
        self.assertRaises(AttributeError, dataframe_functions._normalize_input_to_internal_representation, test_df,
                          "strange_id", "sort", "kind", "value")

        test_df = pd.DataFrame([{"id": 0, "kind": "a", "value": 3, "sort": 1}])
        self.assertRaises(AttributeError, dataframe_functions._normalize_input_to_internal_representation, test_df,
                          "id", "sort", "strange_kind", "value")

        test_df = pd.DataFrame([{"id": np.NaN, "kind": "a", "value": 3, "sort": 1}])
        self.assertRaises(ValueError, dataframe_functions._normalize_input_to_internal_representation, test_df,
                          "id", "sort", "kind", "value")

        test_df = pd.DataFrame([{"id": 0, "kind": np.NaN, "value": 3, "sort": 1}])
        self.assertRaises(ValueError, dataframe_functions._normalize_input_to_internal_representation, test_df,
                          "id", "sort", "kind", "value")

        test_df = pd.DataFrame([{"id": 2}, {"id": 1}])
        test_dict = {"a": test_df, "b": test_df}

        # If there are more than one column, the algorithm can not choose the correct column
        self.assertRaises(ValueError, dataframe_functions._normalize_input_to_internal_representation, test_dict,
                          "id", None, None, None)

        test_dict = {"a": pd.DataFrame([{"id": 2, "value_a": 3}, {"id": 1, "value_a": 4}]),
                     "b": pd.DataFrame([{"id": 2}, {"id": 1}])}

        # If there are more than one column, the algorithm can not choose the correct column
        self.assertRaises(ValueError, dataframe_functions._normalize_input_to_internal_representation, test_dict,
                          "id", None, None, None)

        test_df = pd.DataFrame([{"id": 0, "value": np.NaN}])
        self.assertRaises(ValueError, dataframe_functions._normalize_input_to_internal_representation, test_df,
                          "id", None, None, "value")

        test_df = pd.DataFrame([{"id": 0, "value": np.NaN}])
        self.assertRaises(ValueError, dataframe_functions._normalize_input_to_internal_representation, test_df,
                          None, None, None, "value")

        test_df = pd.DataFrame([{"id": 0, "a_": 3, "b": 5, "sort": 1}])
        self.assertRaises(
            ValueError,
            dataframe_functions._normalize_input_to_internal_representation,
            test_df,
            "id",
            "sort",
            None,
            None)

        test_df = pd.DataFrame([{"id": 0, "a__c": 3, "b": 5, "sort": 1}])
        self.assertRaises(
            ValueError,
            dataframe_functions._normalize_input_to_internal_representation,
            test_df,
            "id",
            "sort",
            None,
            None)

        test_df = pd.DataFrame([{"id": 0}])
        self.assertRaises(ValueError, dataframe_functions._normalize_input_to_internal_representation, test_df,
                          "id", None, None, None)

        test_df = pd.DataFrame([{"id": 0, "sort": 0}])
        self.assertRaises(ValueError, dataframe_functions._normalize_input_to_internal_representation, test_df,
                          "id", "sort", None, None)

    def test_wide_dataframe_order_preserved_with_sort_column(self):
        """ verifies that the order of the sort column from a wide time series container is preserved
        """

        test_df = pd.DataFrame({'id': ["a", "a", "b"],
                                'v1': [3, 2, 1],
                                'v2': [13, 12, 11],
                                'sort': [103, 102, 101]})

        melt_df, _, _, _ = \
            dataframe_functions._normalize_input_to_internal_representation(
                test_df, column_id="id", column_sort="sort", column_kind=None, column_value=None)

        assert (test_df.sort_values("sort").query("id=='a'")["v1"].values ==
                melt_df.query("id=='a'").query("_variables=='v1'")["_values"].values).all()
        assert (test_df.sort_values("sort").query("id=='a'")["v2"].values ==
                melt_df.query("id=='a'").query("_variables=='v2'")["_values"].values).all()

    def test_wide_dataframe_order_preserved(self):
        """ verifies that the order of the time series inside a wide time series container are preserved
        (columns_sort=None)
        """
        test_df = pd.DataFrame({'id': ["a", "a", "a", "b"],
                                'v1': [4, 3, 2, 1],
                                'v2': [14, 13, 12, 11]})

        melt_df, _, _, _ = \
            dataframe_functions._normalize_input_to_internal_representation(
                test_df, column_id="id", column_sort=None, column_kind=None, column_value=None)

        assert (test_df.query("id=='a'")["v1"].values ==
                melt_df.query("id=='a'").query("_variables=='v1'")["_values"].values).all()
        assert (test_df.query("id=='a'")["v2"].values ==
                melt_df.query("id=='a'").query("_variables=='v2'")["_values"].values).all()


class RollingTestCase(TestCase):
    def test_with_wrong_input(self):
        test_df = pd.DataFrame({"id": [0, 0], "kind": ["a", "b"], "value": [3, 3], "sort": [np.NaN, np.NaN]})
        self.assertRaises(ValueError, dataframe_functions.roll_time_series,
                          df_or_dict=test_df, column_id="id",
                          column_sort="sort", column_kind="kind",
                          rolling_direction=1)

        test_df = pd.DataFrame({"id": [0, 0], "kind": ["a", "b"], "value": [3, 3], "sort": [1, 1]})
        self.assertRaises(AttributeError, dataframe_functions.roll_time_series,
                          df_or_dict=test_df, column_id="strange_id",
                          column_sort="sort", column_kind="kind",
                          rolling_direction=1)

        self.assertRaises(ValueError, dataframe_functions.roll_time_series,
                          df_or_dict=test_df, column_id=None,
                          column_sort="sort", column_kind="kind",
                          rolling_direction=1)

        test_df = {"a": pd.DataFrame([{"id": 0}])}
        self.assertRaises(ValueError, dataframe_functions.roll_time_series,
                          df_or_dict=test_df, column_id="id",
                          column_sort=None, column_kind="kind",
                          rolling_direction=1)

        self.assertRaises(ValueError, dataframe_functions.roll_time_series,
                          df_or_dict=test_df, column_id=None,
                          column_sort=None, column_kind="kind",
                          rolling_direction=1)

        self.assertRaises(ValueError, dataframe_functions.roll_time_series,
                          df_or_dict=test_df, column_id="id",
                          column_sort=None, column_kind=None,
                          rolling_direction=0)

        self.assertRaises(ValueError, dataframe_functions.roll_time_series,
                          df_or_dict=test_df, column_id=None,
                          column_sort=None, column_kind=None,
                          rolling_direction=0)

    def test_assert_single_row(self):
        test_df = pd.DataFrame([{"id": np.NaN, "kind": "a", "value": 3, "sort": 1}])
        self.assertRaises(ValueError, dataframe_functions.roll_time_series,
                          df_or_dict=test_df, column_id="id",
                          column_sort="sort", column_kind="kind",
                          rolling_direction=1)

    def test_positive_rolling(self):
        first_class = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "time": range(4)})
        second_class = pd.DataFrame({"a": [10, 11], "b": [12, 13], "time": range(20, 22)})

        first_class["id"] = 1
        second_class["id"] = 2

        df_full = pd.concat([first_class, second_class], ignore_index=True)

        """ df_full is
            a   b  time  id
        0   1   5     0   1
        1   2   6     1   1
        2   3   7     2   1
        3   4   8     3   1
        4  10  12    20   2
        5  11  13    21   2
        """
        correct_indices = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 20, 21, 21]
        correct_values_a = [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 11.0]
        correct_values_b = [5.0, 5.0, 6.0, 5.0, 6.0, 7.0, 5.0, 6.0, 7.0, 8.0, 12.0, 12.0, 13.0]

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=1)

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=1,
                                                  max_timeshift=4)

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=1,
                                                  max_timeshift=2)

        correct_indices = [0, 1, 1, 2, 2, 2, 3, 3, 3, 20, 21, 21]
        correct_values_a = [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 10.0, 10.0, 11.0]
        correct_values_b = [5.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0, 7.0, 8.0, 12.0, 12.0, 13.0]

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

    def test_negative_rolling(self):
        first_class = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "time": range(4)})
        second_class = pd.DataFrame({"a": [10, 11], "b": [12, 13], "time": range(20, 22)})

        first_class["id"] = 1
        second_class["id"] = 2

        df_full = pd.concat([first_class, second_class], ignore_index=True)
        """ df_full is
            a   b  time  id
        0   1   5     0   1
        1   2   6     1   1
        2   3   7     2   1
        3   4   8     3   1
        4  10  12    20   2
        5  11  13    21   2
        """

        correct_indices = ([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 20, 20, 21])
        correct_values_a = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 3.0, 4.0, 4.0, 10.0, 11.0, 11.0]
        correct_values_b = [5.0, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0, 7.0, 8.0, 8.0, 12.0, 13.0, 13.0]

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=-1)

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=-1,
                                                  max_timeshift=None)

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=-1,
                                                  max_timeshift=1)

        correct_indices = ([0, 0, 1, 1, 2, 2, 3, 20, 20, 21])
        correct_values_a = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 10.0, 11.0, 11.0]
        correct_values_b = [5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 12.0, 13.0, 13.0]

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=-1,
                                                  max_timeshift=2)

        correct_indices = ([0, 0, 0, 1, 1, 1, 2, 2, 3, 20, 20, 21])
        correct_values_a = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 4.0, 10.0, 11.0, 11.0]
        correct_values_b = [5.0, 6.0, 7.0, 6.0, 7.0, 8.0, 7.0, 8.0, 8.0, 12.0, 13.0, 13.0]

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=-1,
                                                  max_timeshift=4)

        correct_indices = ([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 20, 20, 21])
        correct_values_a = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 3.0, 4.0, 4.0, 10.0, 11.0, 11.0]
        correct_values_b = [5.0, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0, 7.0, 8.0, 8.0, 12.0, 13.0, 13.0]

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

    def test_stacked_rolling(self):
        first_class = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "time": range(4)})
        second_class = pd.DataFrame({"a": [10, 11], "b": [12, 13], "time": range(20, 22)})

        first_class["id"] = 1
        second_class["id"] = 2

        df_full = pd.concat([first_class, second_class], ignore_index=True)

        df_stacked = pd.concat([df_full[["time", "id", "a"]].rename(columns={"a": "_value"}),
                                df_full[["time", "id", "b"]].rename(columns={"b": "_value"})], ignore_index=True)
        df_stacked["kind"] = ["a"] * 6 + ["b"] * 6

        """ df_stacked is
            time  id  _value kind
        0      0   1       1    a
        1      1   1       2    a
        2      2   1       3    a
        3      3   1       4    a
        4     20   2      10    a
        5     21   2      11    a
        6      0   1       5    b
        7      1   1       6    b
        8      2   1       7    b
        9      3   1       8    b
        10    20   2      12    b
        11    21   2      13    b
        """

        df = dataframe_functions.roll_time_series(df_stacked, column_id="id", column_sort="time",
                                                  column_kind="kind", rolling_direction=-1)

        correct_indices = ([0] * 2 * 4 + [1] * 2 * 3 + [2] * 2 * 2 + [3] * 2 * 1 + [20] * 4 + [21] * 2)
        self.assertListEqual(list(df["id"].values), correct_indices)

        print(df["_value"].values)
        self.assertListEqual(list(df["kind"].values), ["a", "b"] * 13)
        self.assertListEqual(list(df["_value"].values),
                             [1., 5., 2., 6., 3., 7., 4., 8., 2., 6., 3., 7., 4., 8., 3., 7., 4., 8., 4., 8., 10., 12.,
                              11., 13., 11., 13.])

    def test_dict_rolling(self):
        df_dict = {
            "a": pd.DataFrame({"_value": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]}),
            "b": pd.DataFrame({"_value": [5, 6, 7, 8, 12, 13], "id": [1, 1, 1, 1, 2, 2]})
        }
        df = dataframe_functions.roll_time_series(df_dict, column_id="id", column_sort=None, column_kind=None,
                                                  rolling_direction=-1)
        """ df is
        {a: _value  sort id
         7      1.0   0.0  0
         3      2.0   1.0  0
         1      3.0   2.0  0
         0      4.0   3.0  0
         8      2.0   1.0  1
         4      3.0   2.0  1
         2      4.0   3.0  1
         9      3.0   2.0  2
         5      4.0   3.0  2
         10     4.0   3.0  3
         11    10.0   4.0  4
         6     11.0   5.0  4
         12    11.0   5.0  5,

         b: _value  sort id
         7      5.0   0.0  0
         3      6.0   1.0  0
         1      7.0   2.0  0
         0      8.0   3.0  0
         8      6.0   1.0  1
         4      7.0   2.0  1
         2      8.0   3.0  1
         9      7.0   2.0  2
         5      8.0   3.0  2
         10     8.0   3.0  3
         11    12.0   4.0  4
         6     13.0   5.0  4
         12    13.0   5.0  5}
        """

        correct_indices = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5]

        self.assertListEqual(list(df["a"]["id"].values), correct_indices)
        self.assertListEqual(list(df["b"]["id"].values), correct_indices)

        self.assertListEqual(list(df["a"]["_value"].values),
                             [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 3.0, 4.0, 4.0, 10.0, 11.0, 11.0])
        self.assertListEqual(list(df["b"]["_value"].values),
                             [5.0, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0, 7.0, 8.0, 8.0, 12.0, 13.0, 13.0])

    def test_dict_rolling_maxshift_1(self):
        df_dict = {
            "a": pd.DataFrame({"_value": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]}),
            "b": pd.DataFrame({"_value": [5, 6, 7, 8, 12, 13], "id": [1, 1, 1, 1, 2, 2]})
        }
        df = dataframe_functions.roll_time_series(df_dict, column_id="id", column_sort=None, column_kind=None,
                                                  rolling_direction=-1, max_timeshift=1)
        """ df is
        {a: _value  sort id
         7      1.0   0.0  0
         3      2.0   1.0  0
         8      2.0   1.0  1
         4      3.0   2.0  1
         9      3.0   2.0  2
         5      4.0   3.0  2
         10     4.0   3.0  3
         11    10.0   4.0  4
         6     11.0   5.0  4
         12    11.0   5.0  5,

         b: _value  sort id
         7      5.0   0.0  0
         3      6.0   1.0  0
         8      6.0   1.0  1
         4      7.0   2.0  1
         9      7.0   2.0  2
         5      8.0   3.0  2
         10     8.0   3.0  3
         11    12.0   4.0  4
         6     13.0   5.0  4
         12    13.0   5.0  5}
        """

        correct_indices = [0, 0, 1, 1, 2, 2, 3, 4, 4, 5]

        self.assertListEqual(list(df["a"]["id"].values), correct_indices)
        self.assertListEqual(list(df["b"]["id"].values), correct_indices)

        self.assertListEqual(list(df["a"]["_value"].values), [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 10.0, 11.0, 11.0])
        self.assertListEqual(list(df["b"]["_value"].values), [5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 12.0, 13.0, 13.0])

    def test_warning_on_non_uniform_time_steps(self):
        with warnings.catch_warnings(record=True) as w:
            first_class = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "time": [1, 2, 4, 5]})
            second_class = pd.DataFrame({"a": [10, 11], "b": [12, 13], "time": range(20, 22)})

            first_class["id"] = 1
            second_class["id"] = 2

            df_full = pd.concat([first_class, second_class], ignore_index=True)

            dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                 column_kind=None, rolling_direction=1)

            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[0].message),
                             "Your time stamps are not uniformly sampled, which makes rolling "
                             "nonsensical in some domains.")


class CheckForNanTestCase(TestCase):
    def test_all_columns(self):
        test_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[0, 1])

        # should not raise an exception
        dataframe_functions.check_for_nans_in_columns(test_df)

        test_df = pd.DataFrame([[1, 2, 3], [4, np.NaN, 6]], index=[0, 1])

        self.assertRaises(ValueError, dataframe_functions.check_for_nans_in_columns, test_df)

    def test_not_all_columns(self):
        test_df = pd.DataFrame([[1, 2, 3], [4, np.NaN, 6]], index=[0, 1], columns=["a", "b", "c"])

        self.assertRaises(ValueError, dataframe_functions.check_for_nans_in_columns, test_df)
        self.assertRaises(ValueError, dataframe_functions.check_for_nans_in_columns, test_df, ["a", "b"])
        self.assertRaises(ValueError, dataframe_functions.check_for_nans_in_columns, test_df, ["b"])
        self.assertRaises(ValueError, dataframe_functions.check_for_nans_in_columns, test_df, "b")
        self.assertRaises(ValueError, dataframe_functions.check_for_nans_in_columns, test_df, ["c", "b"])

        dataframe_functions.check_for_nans_in_columns(test_df, columns=["a", "c"])
        dataframe_functions.check_for_nans_in_columns(test_df, columns="a")


class ImputeTestCase(TestCase):
    def test_impute_zero(self):
        df = pd.DataFrame([{"value": np.NaN}])
        dataframe_functions.impute_dataframe_zero(df)
        self.assertEqual(list(df.value), [0])

        df = pd.DataFrame([{"value": np.PINF}])
        dataframe_functions.impute_dataframe_zero(df)
        self.assertEqual(list(df.value), [0])

        df = pd.DataFrame([{"value": np.NINF}])
        dataframe_functions.impute_dataframe_zero(df)
        self.assertEqual(list(df.value), [0])

        df = pd.DataFrame([{"value": np.NINF}, {"value": np.NaN}, {"value": np.PINF}, {"value": 1}])
        dataframe_functions.impute_dataframe_zero(df)
        self.assertEqual(list(df.value), [0, 0, 0, 1])

        df = pd.DataFrame([{"value": np.NINF}, {"value": np.NaN}, {"value": np.PINF}, {"value": 1}])
        df = df.astype(np.float64)
        df = dataframe_functions.impute_dataframe_zero(df)
        self.assertEqual(list(df.value), [0, 0, 0, 1])

        df = pd.DataFrame([{"value": np.NINF}, {"value": np.NaN}, {"value": np.PINF}, {"value": 1}])
        df = df.astype(np.float32)
        df = dataframe_functions.impute_dataframe_zero(df)
        self.assertEqual(list(df.value), [0, 0, 0, 1])

    def test_toplevel_impute(self):
        df = pd.DataFrame(np.transpose([[0, 1, 2, np.NaN], [1, np.PINF, 2, 3], [1, -3, np.NINF, 3]]),
                          columns=["value_a", "value_b", "value_c"])

        dataframe_functions.impute(df)

        self.assertEqual(list(df.value_a), [0, 1, 2, 1])
        self.assertEqual(list(df.value_b), [1, 3, 2, 3])
        self.assertEqual(list(df.value_c), [1, -3, -3, 3])

        df = pd.DataFrame(np.transpose([[0, 1, 2, np.NaN], [1, np.PINF, 2, np.NaN], [np.NaN, -3, np.NINF, 3]]),
                          columns=["value_a", "value_b", "value_c"])
        df = df.astype(np.float64)
        dataframe_functions.impute(df)

        self.assertEqual(list(df.value_a), [0, 1, 2, 1])
        self.assertEqual(list(df.value_b), [1, 2, 2, 1.5])
        self.assertEqual(list(df.value_c), [0, -3, -3, 3])

        df = pd.DataFrame(np.transpose([[0, 1, 2, np.NaN], [1, np.PINF, 2, 3], [np.PINF, -3, np.NINF, 3]]),
                          columns=["value_a", "value_b", "value_c"])
        df = df.astype(np.float32)
        dataframe_functions.impute(df)

        self.assertEqual(list(df.value_a), [0, 1, 2, 1])
        self.assertEqual(list(df.value_b), [1, 3, 2, 3])
        self.assertEqual(list(df.value_c), [3, -3, -3, 3])

    def test_impute_range(self):
        def get_df():
            return pd.DataFrame(np.transpose([[0, 1, 2, np.NaN],
                                              [1, np.PINF, 2, 3],
                                              [1, -3, np.NINF, 3]]),
                                columns=["value_a", "value_b", "value_c"])

        # check if values are replaced correctly
        df = get_df()
        col_to_max = {"value_a": 200, "value_b": 200, "value_c": 200}
        col_to_min = {"value_a": -134, "value_b": -134, "value_c": -134}
        col_to_median = {"value_a": 55, "value_b": 55, "value_c": 55}
        dataframe_functions.impute_dataframe_range(df, col_to_max, col_to_min, col_to_median)
        self.assertEqual(list(df.value_a), [0, 1, 2, 55])
        self.assertEqual(list(df.value_b), [1, 200, 2, 3])
        self.assertEqual(list(df.value_c), [1, -3, -134, 3])

        # check for error if column key is missing
        df = get_df()
        col_to_max = {"value_a": 200, "value_b": 200, "value_c": 200}
        col_to_min = {"value_a": -134, "value_b": -134, "value_c": -134}
        col_to_median = {"value_a": 55, "value_c": 55}
        self.assertRaises(ValueError, dataframe_functions.impute_dataframe_range,
                          df, col_to_max, col_to_min, col_to_median)

        # check for no error if column key is too much
        col_to_max = {"value_a": 200, "value_b": 200, "value_c": 200}
        col_to_min = {"value_a": -134, "value_b": -134, "value_c": -134}
        col_to_median = {"value_a": 55, "value_b": 55, "value_c": 55, "value_d": 55}
        dataframe_functions.impute_dataframe_range(df, col_to_max, col_to_min, col_to_median)

        # check for error if replacement value is not finite
        df = get_df()
        col_to_max = {"value_a": 200, "value_b": np.NaN, "value_c": 200}
        col_to_min = {"value_a": -134, "value_b": -134, "value_c": -134}
        col_to_median = {"value_a": 55, "value_b": 55, "value_c": 55}
        self.assertRaises(ValueError, dataframe_functions.impute_dataframe_range,
                          df, col_to_max, col_to_min, col_to_median)
        df = get_df()
        col_to_max = {"value_a": 200, "value_b": 200, "value_c": 200}
        col_to_min = {"value_a": -134, "value_b": np.NINF, "value_c": -134}
        col_to_median = {"value_a": 55, "value_b": 55, "value_c": 55}
        self.assertRaises(ValueError, dataframe_functions.impute_dataframe_range,
                          df, col_to_max, col_to_min, col_to_median)

        df = get_df()
        col_to_max = {"value_a": 200, "value_b": 200, "value_c": 200}
        col_to_min = {"value_a": -134, "value_b": -134, "value_c": -134}
        col_to_median = {"value_a": 55, "value_b": 55, "value_c": np.PINF}
        self.assertRaises(ValueError, dataframe_functions.impute_dataframe_range,
                          df, col_to_max, col_to_min, col_to_median)

        df = pd.DataFrame([0, 1, 2, 3, 4], columns=["test"])
        col_dict = {"test": 0}
        dataframe_functions.impute_dataframe_range(df, col_dict, col_dict, col_dict)

        self.assertEqual(df.columns, ["test"])
        self.assertListEqual(list(df["test"].values), [0, 1, 2, 3, 4])


class RestrictTestCase(TestCase):
    def test_restrict_dataframe(self):
        df = pd.DataFrame({'id': [1, 2, 3] * 2})

        df_restricted = dataframe_functions.restrict_input_to_index(df, 'id', [2])
        self.assertEqual(list(df_restricted.id), [2, 2])

        df_restricted2 = dataframe_functions.restrict_input_to_index(df, 'id', [1, 2, 3])
        self.assertTrue(df_restricted2.equals(df))

    def test_restrict_dict(self):
        kind_to_df = {'a': pd.DataFrame({'id': [1, 2, 3]}), 'b': pd.DataFrame({'id': [3, 4, 5]})}

        kind_to_df_restricted = dataframe_functions.restrict_input_to_index(kind_to_df, 'id', [3])
        self.assertEqual(list(kind_to_df_restricted['a'].id), [3])
        self.assertEqual(list(kind_to_df_restricted['b'].id), [3])

        kind_to_df_restricted2 = dataframe_functions.restrict_input_to_index(kind_to_df, 'id', [1, 2, 3, 4, 5])
        self.assertTrue(kind_to_df_restricted2['a'].equals(kind_to_df['a']))
        self.assertTrue(kind_to_df_restricted2['b'].equals(kind_to_df['b']))

    def test_restrict_wrong(self):
        other_type = np.array([1, 2, 3])

        self.assertRaises(TypeError, dataframe_functions.restrict_input_to_index, other_type, "id", [1, 2, 3])


class GetRangeValuesPerColumnTestCase(TestCase):
    def test_ignores_non_finite_values(self):
        df = pd.DataFrame([0, 1, 2, 3, np.NaN, np.PINF, np.NINF], columns=["value"])

        col_to_max, col_to_min, col_to_median = dataframe_functions.get_range_values_per_column(df)

        self.assertEqual(col_to_max, {"value": 3})
        self.assertEqual(col_to_min, {"value": 0})
        self.assertEqual(col_to_median, {"value": 1.5})

    def test_range_values_correct_with_even_length(self):
        df = pd.DataFrame([0, 1, 2, 3], columns=["value"])

        col_to_max, col_to_min, col_to_median = dataframe_functions.get_range_values_per_column(df)

        self.assertEqual(col_to_max, {"value": 3})
        self.assertEqual(col_to_min, {"value": 0})
        self.assertEqual(col_to_median, {"value": 1.5})

    def test_range_values_correct_with_uneven_length(self):
        df = pd.DataFrame([0, 1, 2], columns=["value"])

        col_to_max, col_to_min, col_to_median = dataframe_functions.get_range_values_per_column(df)

        self.assertEqual(col_to_max, {"value": 2})
        self.assertEqual(col_to_min, {"value": 0})
        self.assertEqual(col_to_median, {"value": 1})

    def test_no_finite_values_yields_0(self):
        df = pd.DataFrame([np.NaN, np.PINF, np.NINF], columns=["value"])

        col_to_max, col_to_min, col_to_median = dataframe_functions.get_range_values_per_column(df)

        self.assertEqual(col_to_max, {"value": 0})
        self.assertEqual(col_to_min, {"value": 0})
        self.assertEqual(col_to_median, {"value": 0})


class MakeForecastingFrameTestCase(TestCase):

    def test_make_forecasting_frame_list(self):
        df, y = dataframe_functions.make_forecasting_frame(x=range(4), kind="test",
                                                           max_timeshift=1, rolling_direction=1)
        expected_df = pd.DataFrame({"id": [1, 2, 3], "kind": ["test"] * 3, "value": [0., 1., 2.], "time": [0., 1., 2.]})

        expected_y = pd.Series(data=[1, 2, 3], index=[1, 2, 3], name="value")
        assert_frame_equal(df.sort_index(axis=1), expected_df.sort_index(axis=1))
        assert_series_equal(y, expected_y)

    def test_make_forecasting_frame_range(self):
        df, y = dataframe_functions.make_forecasting_frame(x=np.arange(4), kind="test",
                                                           max_timeshift=1, rolling_direction=1)
        expected_df = pd.DataFrame({"id": [1, 2, 3], "kind": ["test"] * 3, "value": [0., 1., 2.], "time": [0., 1., 2.]})
        assert_frame_equal(df.sort_index(axis=1), expected_df.sort_index(axis=1))

    def test_make_forecasting_frame_pdSeries(self):

        t_index = pd.date_range('1/1/2011', periods=4, freq='H')
        df, y = dataframe_functions.make_forecasting_frame(x=pd.Series(data=range(4), index=t_index),
                                                           kind="test", max_timeshift=1, rolling_direction=1)

        expected_y = pd.Series(data=[1, 2, 3], index=pd.DatetimeIndex(["2011-01-01 01:00:00", "2011-01-01 02:00:00",
                                                                       "2011-01-01 03:00:00"]), name="value")
        expected_df = pd.DataFrame({"id": pd.DatetimeIndex(["2011-01-01 01:00:00", "2011-01-01 02:00:00",
                                                            "2011-01-01 03:00:00"]),
                                    "kind": ["test"] * 3, "value": [0., 1., 2.],
                                    "time": pd.DatetimeIndex(["2011-01-01 00:00:00", "2011-01-01 01:00:00",
                                                              "2011-01-01 02:00:00"])
                                    })
        assert_frame_equal(df.sort_index(axis=1), expected_df.sort_index(axis=1))
        assert_series_equal(y, expected_y)


class GetIDsTestCase(TestCase):

    def test_get_id__correct_DataFrame(self):
        df = pd.DataFrame({"_value": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]})
        self.assertEqual(get_ids(df, "id"), {1, 2})

    def test_get_id__correct_dict(self):
        df_dict = {"a": pd.DataFrame({"_value": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]}),
                   "b": pd.DataFrame({"_value": [5, 6, 7, 8, 12, 13], "id": [4, 4, 3, 3, 2, 2]})}
        self.assertEqual(get_ids(df_dict, "id"), {1, 2, 3, 4})

    def test_get_id_wrong(self):
        other_type = np.array([1, 2, 3])
        self.assertRaises(TypeError, get_ids, other_type, "id")
