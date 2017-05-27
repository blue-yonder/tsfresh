# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
import warnings
from unittest import TestCase

import pandas as pd
from tsfresh.utilities import dataframe_functions
import numpy as np
import six


class NormalizeTestCase(TestCase):
    def test_with_dictionaries_one_row(self):
        test_df = pd.DataFrame([{"value": 1, "id": "id_1"}])
        test_dict = {"a": test_df, "b": test_df}

        # A kind is not allowed with dicts
        self.assertRaises(ValueError, dataframe_functions.normalize_input_to_internal_representation, test_dict,
                          "id", None, "a kind", None)

        # The value must be present
        self.assertRaises(ValueError, dataframe_functions.normalize_input_to_internal_representation, test_dict,
                          "id", None, None, "something other")

        # Nothing should have changed compared to the input data
        result_dict, column_id, column_value = \
            dataframe_functions.normalize_input_to_internal_representation(test_dict, "id", None, None, "value")
        self.assertEqual(column_value, "value")
        self.assertEqual(column_id, "id")
        six.assertCountEqual(self, list(test_dict.keys()), list(result_dict.keys()))
        self.assertEqual(result_dict["a"].iloc[0].to_dict(), {"value": 1, "id": "id_1"})

        # The algo should choose the correct value column
        result_dict, column_id, column_value = \
            dataframe_functions.normalize_input_to_internal_representation(test_dict, "id", None, None, None)
        self.assertEqual(column_value, "value")
        self.assertEqual(column_id, "id")

    def test_with_dictionaries_two_rows(self):
        test_df = pd.DataFrame([{"value": 2, "sort": 2, "id": "id_1"},
                                {"value": 1, "sort": 1, "id": "id_1"}])
        test_dict = {"a": test_df, "b": test_df}

        # If there are more than one column, the algorithm can not choose the correct column
        self.assertRaises(ValueError, dataframe_functions.normalize_input_to_internal_representation, test_dict,
                          "id", None, None, None)

        # Sorting should work
        result_dict, column_id, column_value = \
            dataframe_functions.normalize_input_to_internal_representation(test_dict, "id", "sort", None, "value")
        self.assertEqual(column_value, "value")
        self.assertEqual(column_id, "id")

        # Assert sorted and without sort column
        self.assertEqual(result_dict["a"].iloc[0].to_dict(), {"value": 1, "id": "id_1"})
        self.assertEqual(result_dict["a"].iloc[1].to_dict(), {"value": 2, "id": "id_1"})

        # Assert the algo has found the correct column
        result_dict, column_id, column_value = \
            dataframe_functions.normalize_input_to_internal_representation(test_dict, "id", "sort", None, None)
        self.assertEqual(column_value, "value")
        self.assertEqual(column_id, "id")

    def test_with_dictionaries_two_rows_sorted(self):
        test_df = pd.DataFrame([{"value": 2, "id": "id_1"},
                                {"value": 1, "id": "id_1"}])
        test_dict = {"a": test_df, "b": test_df}

        # Pass the id
        result_dict, column_id, column_value = \
            dataframe_functions.normalize_input_to_internal_representation(test_dict, "id", None, None, "value")
        self.assertEqual(column_value, "value")
        self.assertEqual(column_id, "id")

        self.assertEqual(result_dict["a"].iloc[0].to_dict(), {"value": 2, "id": "id_1"})

        # The algo should have found the correct value column
        result_dict, column_id, column_value = \
            dataframe_functions.normalize_input_to_internal_representation(test_dict, "id", None, None, None)
        self.assertEqual(column_value, "value")
        self.assertEqual(column_id, "id")

    def test_with_df(self):
        # give everyting
        test_df = pd.DataFrame([{"id": 0, "kind": "a", "value": 3, "sort": 1}])
        result_dict, column_id, column_value = \
            dataframe_functions.normalize_input_to_internal_representation(test_df, "id", "sort", "kind", "value")

        self.assertEqual(column_id, "id")
        self.assertEqual(column_value, "value")
        self.assertIn("a", result_dict)
        six.assertCountEqual(self, list(result_dict["a"].columns), ["id", "value"])
        self.assertEqual(list(result_dict["a"]["value"]), [3])
        self.assertEqual(list(result_dict["a"]["id"]), [0])

        # give no kind
        test_df = pd.DataFrame([{"id": 0, "value": 3, "sort": 1}])
        result_dict, column_id, column_value = \
            dataframe_functions.normalize_input_to_internal_representation(test_df, "id", "sort", None, "value")

        self.assertEqual(column_id, "id")
        self.assertEqual(column_value, "value")
        self.assertIn("value", result_dict)
        six.assertCountEqual(self, list(result_dict["value"].columns), ["id", "value"])
        self.assertEqual(list(result_dict["value"]["value"]), [3])
        self.assertEqual(list(result_dict["value"]["id"]), [0])

        # Let the function find the values
        test_df = pd.DataFrame([{"id": 0, "a": 3, "b": 5, "sort": 1}])
        result_dict, column_id, column_value = \
            dataframe_functions.normalize_input_to_internal_representation(test_df, "id", "sort", None, None)

        self.assertEqual(column_id, "id")
        self.assertEqual(column_value, "_value")
        self.assertIn("a", result_dict)
        self.assertIn("b", result_dict)
        six.assertCountEqual(self, list(result_dict["a"].columns), ["_value", "id"])
        self.assertEqual(list(result_dict["a"]["_value"]), [3])
        self.assertEqual(list(result_dict["a"]["id"]), [0])
        six.assertCountEqual(self, list(result_dict["b"].columns), ["_value", "id"])
        self.assertEqual(list(result_dict["b"]["_value"]), [5])
        self.assertEqual(list(result_dict["b"]["id"]), [0])

    def test_with_wrong_input(self):
        test_df = pd.DataFrame([{"id": 0, "kind": "a", "value": 3, "sort": np.NaN}])
        self.assertRaises(ValueError, dataframe_functions.normalize_input_to_internal_representation, test_df,
                          "id", "sort", "kind", "value")

        test_df = pd.DataFrame([{"id": 0, "kind": "a", "value": 3, "sort": 1}])
        self.assertRaises(AttributeError, dataframe_functions.normalize_input_to_internal_representation, test_df,
                          "strange_id", "sort", "kind", "value")

        test_df = pd.DataFrame([{"id": np.NaN, "kind": "a", "value": 3, "sort": 1}])
        self.assertRaises(ValueError, dataframe_functions.normalize_input_to_internal_representation, test_df,
                          "id", "sort", "kind", "value")

        test_df = pd.DataFrame([{"id": 0}])
        self.assertRaises(ValueError, dataframe_functions.normalize_input_to_internal_representation, test_df,
                          "id", None, None, None)

        test_df = pd.DataFrame([{"id": 2}, {"id": 1}])
        test_dict = {"a": test_df, "b": test_df}

        # If there are more than one column, the algorithm can not choose the correct column
        self.assertRaises(ValueError, dataframe_functions.normalize_input_to_internal_representation, test_dict,
                          "id", None, None, None)

        test_dict = {"a": pd.DataFrame([{"id": 2, "value_a": 3}, {"id": 1, "value_a": 4}]),
                     "b": pd.DataFrame([{"id": 2}, {"id": 1}])}

        # If there are more than one column, the algorithm can not choose the correct column
        self.assertRaises(ValueError, dataframe_functions.normalize_input_to_internal_representation, test_dict,
                          "id", None, None, None)

        test_df = pd.DataFrame([{"id": 0, "value": np.NaN}])
        self.assertRaises(ValueError, dataframe_functions.normalize_input_to_internal_representation, test_df,
                          "id", None, None, "value")

        test_df = pd.DataFrame([{"id": 0, "value": np.NaN}])
        self.assertRaises(ValueError, dataframe_functions.normalize_input_to_internal_representation, test_df,
                          None, None, None, "value")


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

        correct_indices = (["id=1, shift=3"] * 1 +
                           ["id=1, shift=2"] * 2 +
                           ["id=1, shift=1"] * 3 +
                           ["id=2, shift=1"] * 1 +
                           ["id=1, shift=0"] * 4 +
                           ["id=2, shift=0"] * 2)
        correct_values_a = [1, 1, 2, 1, 2, 3, 10, 1, 2, 3, 4, 10, 11]
        correct_values_b = [5, 5, 6, 5, 6, 7, 12, 5, 6, 7, 8, 12, 13]

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=1)

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=1,
                                                  maximum_number_of_timeshifts=None)

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=1,
                                                  maximum_number_of_timeshifts=1)

        self.assertListEqual(list(df["id"]), correct_indices[3:])
        self.assertListEqual(list(df["a"].values), correct_values_a[3:])
        self.assertListEqual(list(df["b"].values), correct_values_b[3:])

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=1,
                                                  maximum_number_of_timeshifts=2)

        self.assertListEqual(list(df["id"]), correct_indices[1:])
        self.assertListEqual(list(df["a"].values), correct_values_a[1:])
        self.assertListEqual(list(df["b"].values), correct_values_b[1:])

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=1,
                                                  maximum_number_of_timeshifts=4)

        self.assertListEqual(list(df["id"]), correct_indices[:])
        self.assertListEqual(list(df["a"].values), correct_values_a[:])
        self.assertListEqual(list(df["b"].values), correct_values_b[:])

    def test_negative_rolling(self):
        first_class = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "time": range(4)})
        second_class = pd.DataFrame({"a": [10, 11], "b": [12, 13], "time": range(20, 22)})

        first_class["id"] = 1
        second_class["id"] = 2

        df_full = pd.concat([first_class, second_class], ignore_index=True)

        correct_indices = (["id=1, shift=-3"] * 1 +
                           ["id=1, shift=-2"] * 2 +
                           ["id=1, shift=-1"] * 3 +
                           ["id=2, shift=-1"] * 1 +
                           ["id=1, shift=0"] * 4 +
                           ["id=2, shift=0"] * 2)
        correct_values_a = [4, 3, 4, 2, 3, 4, 11, 1, 2, 3, 4, 10, 11]
        correct_values_b = [8, 7, 8, 6, 7, 8, 13, 5, 6, 7, 8, 12, 13]

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=-1)

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=-1,
                                                  maximum_number_of_timeshifts=None)

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=-1,
                                                  maximum_number_of_timeshifts=1)

        self.assertListEqual(list(df["id"].values), correct_indices[3:])
        self.assertListEqual(list(df["a"].values), correct_values_a[3:])
        self.assertListEqual(list(df["b"].values), correct_values_b[3:])

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=-1,
                                                  maximum_number_of_timeshifts=2)

        self.assertListEqual(list(df["id"].values), correct_indices[1:])
        self.assertListEqual(list(df["a"].values), correct_values_a[1:])
        self.assertListEqual(list(df["b"].values), correct_values_b[1:])

        df = dataframe_functions.roll_time_series(df_full, column_id="id", column_sort="time",
                                                  column_kind=None, rolling_direction=-1,
                                                  maximum_number_of_timeshifts=4)

        self.assertListEqual(list(df["id"].values), correct_indices[:])
        self.assertListEqual(list(df["a"].values), correct_values_a[:])
        self.assertListEqual(list(df["b"].values), correct_values_b[:])

    def test_stacked_rolling(self):
        first_class = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "time": range(4)})
        second_class = pd.DataFrame({"a": [10, 11], "b": [12, 13], "time": range(20, 22)})

        first_class["id"] = 1
        second_class["id"] = 2

        df_full = pd.concat([first_class, second_class], ignore_index=True)

        df_stacked = pd.concat([df_full[["time", "id", "a"]].rename(columns={"a": "_value"}),
                                df_full[["time", "id", "b"]].rename(columns={"b": "_value"})], ignore_index=True)
        df_stacked["kind"] = ["a"] * 6 + ["b"] * 6

        df = dataframe_functions.roll_time_series(df_stacked, column_id="id", column_sort="time",
                                                  column_kind="kind", rolling_direction=-1)

        correct_indices = (["id=1, shift=-3"] * 2 +
                           ["id=1, shift=-2"] * 4 +
                           ["id=1, shift=-1"] * 6 +
                           ["id=2, shift=-1"] * 2 +
                           ["id=1, shift=0"] * 8 +
                           ["id=2, shift=0"] * 4)

        self.assertListEqual(list(df["id"].values), correct_indices)

        self.assertListEqual(list(df["kind"].values), ["a", "b"] * 13)
        self.assertListEqual(list(df["_value"].values),
                             [4, 8, 3, 7, 4, 8, 2, 6, 3, 7, 4, 8, 11, 13, 1, 5, 2, 6, 3, 7, 4, 8, 10, 12, 11, 13])

    def test_dict_rolling(self):
        df_dict = {
            "a": pd.DataFrame({"_value": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]}),
            "b": pd.DataFrame({"_value": [5, 6, 7, 8, 12, 13], "id": [1, 1, 1, 1, 2, 2]})
        }

        df = dataframe_functions.roll_time_series(df_dict, column_id="id", column_sort=None,
                                                  column_kind=None, rolling_direction=-1)

        correct_indices = (["id=1, shift=-3"] * 1 +
                           ["id=1, shift=-2"] * 2 +
                           ["id=1, shift=-1"] * 3 +
                           ["id=2, shift=-1"] * 1 +
                           ["id=1, shift=0"] * 4 +
                           ["id=2, shift=0"] * 2)

        self.assertListEqual(list(df["a"]["id"].values), correct_indices)
        self.assertListEqual(list(df["b"]["id"].values), correct_indices)

        self.assertListEqual(list(df["a"]["_value"].values),
                             [4, 3, 4, 2, 3, 4, 11, 1, 2, 3, 4, 10, 11])
        self.assertListEqual(list(df["b"]["_value"].values),
                             [8, 7, 8, 6, 7, 8, 13, 5, 6, 7, 8, 12, 13])

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

        df = pd.DataFrame(np.transpose([[0, 1, 2, np.NaN], [1, np.PINF, 2, 3], [1, -3, np.NINF, 3]]),
                          columns=["value_a", "value_b", "value_c"])
        df = df.astype(np.float64, inplace=True)
        dataframe_functions.impute(df)

        df = pd.DataFrame(np.transpose([[0, 1, 2, np.NaN], [1, np.PINF, 2, 3], [1, -3, np.NINF, 3]]),
                          columns=["value_a", "value_b", "value_c"])
        df = df.astype(np.float32, inplace=True)
        dataframe_functions.impute(df)

        self.assertEqual(list(df.value_a), [0, 1, 2, 1])
        self.assertEqual(list(df.value_b), [1, 3, 2, 3])
        self.assertEqual(list(df.value_c), [1, -3, -3, 3])

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
