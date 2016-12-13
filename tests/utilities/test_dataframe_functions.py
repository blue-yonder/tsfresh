# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

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
        result_dict, column_id, column_value =\
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
        self.assertRaises(ValueError, dataframe_functions.check_for_nans_in_columns, test_df, ["c", "b"])

        dataframe_functions.check_for_nans_in_columns(test_df, columns=["a", "c"])


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

    def test_get_range_values(self):
        df = pd.DataFrame([0, 1, 2, 3, np.NaN], columns=["value"])

        col_to_max, col_to_min, col_to_median = dataframe_functions.get_range_values_per_column(df)

        self.assertEqual(col_to_max, {"value": 3})
        self.assertEqual(col_to_min, {"value": 0})
        self.assertEqual(col_to_median, {"value": 1.5})

        df = pd.DataFrame([np.NaN, np.NaN], columns=["value"])

        col_to_max, col_to_min, col_to_median = dataframe_functions.get_range_values_per_column(df)

        self.assertEqual(col_to_max, {"value": 0})
        self.assertEqual(col_to_min, {"value": 0})
        self.assertEqual(col_to_median, {"value": 0})

    def test_toplevel_impute(self):
        df = pd.DataFrame(np.transpose([[0, 1, 2, np.NaN], [1, np.PINF, 2, 3], [1, -3, np.NINF, 3]]),
                          columns=["value_a", "value_b", "value_c"])

        dataframe_functions.impute(df)

        self.assertEqual(list(df.value_a), [0, 1, 2, 1])
        self.assertEqual(list(df.value_b), [1, 3, 2, 3])
        self.assertEqual(list(df.value_c), [1, -3, -3, 3])

    def test_impute_range(self):
        df = pd.DataFrame(np.transpose([[0, 1, 2, np.NaN], [1, np.PINF, 2, 3], [1, -3, np.NINF, 3]]),
                          columns=["value_a", "value_b", "value_c"])

        col_to_max = {"value_b": 200}
        col_to_min = {"value_c": -134}
        col_to_median = {"value_a": 55}

        dataframe_functions.impute_dataframe_range(df, col_to_max, col_to_min, col_to_median)

        self.assertEqual(list(df.value_a), [0, 1, 2, 55])
        self.assertEqual(list(df.value_b), [1, 200, 2, 3])
        self.assertEqual(list(df.value_c), [1, -3, -134, 3])

        df = pd.DataFrame(np.transpose([[0, 1, 2, np.NaN], [1, np.PINF, 2, 3], [1, -3, np.NINF, 3]]),
                          columns=["value_a", "value_b", "value_c"])

        dataframe_functions.impute_dataframe_range(df)

        self.assertEqual(list(df.value_a), [0, 1, 2, 1])
        self.assertEqual(list(df.value_b), [1, 3, 2, 3])
        self.assertEqual(list(df.value_c), [1, -3, -3, 3])

        df = pd.DataFrame(np.transpose([[np.NaN, np.NaN, np.NaN, np.NaN], [1, np.PINF, 2, 3], [1, -3, np.NINF, 3]]),
                          columns=["value_a", "value_b", "value_c"])

        dataframe_functions.impute_dataframe_range(df)

        self.assertEqual(list(df.value_a), [0, 0, 0, 0])
        self.assertEqual(list(df.value_b), [1, 3, 2, 3])
        self.assertEqual(list(df.value_c), [1, -3, -3, 3])


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
