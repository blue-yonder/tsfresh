# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
import warnings
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from tests.fixtures import warning_free
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
from tsfresh.utilities import dataframe_functions


class RollingTestCase(TestCase):
    def test_with_wrong_input(self):
        test_df = pd.DataFrame(
            {
                "id": [0, 0],
                "kind": ["a", "b"],
                "value": [3, 3],
                "sort": [np.NaN, np.NaN],
            }
        )
        self.assertRaises(
            ValueError,
            dataframe_functions.roll_time_series,
            df_or_dict=test_df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            rolling_direction=1,
            n_jobs=0,
        )

        test_df = pd.DataFrame(
            {"id": [0, 0], "kind": ["a", "b"], "value": [3, 3], "sort": [1, 1]}
        )
        self.assertRaises(
            AttributeError,
            dataframe_functions.roll_time_series,
            df_or_dict=test_df,
            column_id="strange_id",
            column_sort="sort",
            column_kind="kind",
            rolling_direction=1,
            n_jobs=0,
        )

        self.assertRaises(
            ValueError,
            dataframe_functions.roll_time_series,
            df_or_dict=test_df,
            column_id=None,
            column_sort="sort",
            column_kind="kind",
            rolling_direction=1,
            n_jobs=0,
        )

        test_df = {"a": pd.DataFrame([{"id": 0}])}
        self.assertRaises(
            ValueError,
            dataframe_functions.roll_time_series,
            df_or_dict=test_df,
            column_id="id",
            column_sort=None,
            column_kind="kind",
            rolling_direction=1,
            n_jobs=0,
        )

        self.assertRaises(
            ValueError,
            dataframe_functions.roll_time_series,
            df_or_dict=test_df,
            column_id=None,
            column_sort=None,
            column_kind="kind",
            rolling_direction=1,
            n_jobs=0,
        )

        self.assertRaises(
            ValueError,
            dataframe_functions.roll_time_series,
            df_or_dict=test_df,
            column_id="id",
            column_sort=None,
            column_kind=None,
            rolling_direction=0,
            n_jobs=0,
        )

        self.assertRaises(
            ValueError,
            dataframe_functions.roll_time_series,
            df_or_dict=test_df,
            column_id=None,
            column_sort=None,
            column_kind=None,
            rolling_direction=0,
            n_jobs=0,
        )

        test_df = pd.DataFrame(
            {"id": [0, 0], "kind": ["a", "b"], "value": [3, 3], "sort": [1, 1]}
        )
        self.assertRaises(
            ValueError,
            dataframe_functions.roll_time_series,
            df_or_dict=test_df,
            column_id="id",
            column_kind="kind",
            column_sort="sort",
            max_timeshift=0,
            rolling_direction=1,
            n_jobs=0,
        )

        self.assertRaises(
            ValueError,
            dataframe_functions.roll_time_series,
            df_or_dict=test_df,
            column_id="id",
            column_kind="kind",
            column_sort="sort",
            min_timeshift=-1,
            rolling_direction=1,
            n_jobs=0,
        )

    def test_assert_single_row(self):
        test_df = pd.DataFrame([{"id": np.NaN, "kind": "a", "value": 3, "sort": 1}])
        self.assertRaises(
            ValueError,
            dataframe_functions.roll_time_series,
            df_or_dict=test_df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            rolling_direction=1,
            n_jobs=0,
        )

    def test_positive_rolling(self):
        first_class = pd.DataFrame(
            {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "time": range(4)}
        )
        second_class = pd.DataFrame(
            {"a": [10, 11], "b": [12, 13], "time": range(20, 22)}
        )

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
        correct_indices = [
            (1, 0),
            (1, 1),
            (1, 1),
            (1, 2),
            (1, 2),
            (1, 2),
            (1, 3),
            (1, 3),
            (1, 3),
            (1, 3),
            (2, 20),
            (2, 21),
            (2, 21),
        ]
        correct_values_a = [
            1.0,
            1.0,
            2.0,
            1.0,
            2.0,
            3.0,
            1.0,
            2.0,
            3.0,
            4.0,
            10.0,
            10.0,
            11.0,
        ]
        correct_values_b = [
            5.0,
            5.0,
            6.0,
            5.0,
            6.0,
            7.0,
            5.0,
            6.0,
            7.0,
            8.0,
            12.0,
            12.0,
            13.0,
        ]

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=1,
            n_jobs=0,
        )

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=1,
            max_timeshift=4,
            n_jobs=0,
        )

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=1,
            max_timeshift=2,
            n_jobs=0,
        )
        correct_indices = [
            (1, 0),
            (1, 1),
            (1, 1),
            (1, 2),
            (1, 2),
            (1, 2),
            (1, 3),
            (1, 3),
            (1, 3),
            (2, 20),
            (2, 21),
            (2, 21),
        ]
        correct_values_a = [
            1.0,
            1.0,
            2.0,
            1.0,
            2.0,
            3.0,
            2.0,
            3.0,
            4.0,
            10.0,
            10.0,
            11.0,
        ]
        correct_values_b = [
            5.0,
            5.0,
            6.0,
            5.0,
            6.0,
            7.0,
            6.0,
            7.0,
            8.0,
            12.0,
            12.0,
            13.0,
        ]

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=1,
            max_timeshift=2,
            min_timeshift=2,
            n_jobs=0,
        )

        correct_indices = [(1, 2), (1, 2), (1, 2), (1, 3), (1, 3), (1, 3)]
        correct_values_a = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]
        correct_values_b = [5.0, 6.0, 7.0, 6.0, 7.0, 8.0]

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

    def test_negative_rolling(self):
        first_class = pd.DataFrame(
            {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "time": range(4)}
        )
        second_class = pd.DataFrame(
            {"a": [10, 11], "b": [12, 13], "time": range(20, 22)}
        )

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

        correct_indices = [
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 2),
            (1, 2),
            (1, 3),
            (2, 20),
            (2, 20),
            (2, 21),
        ]
        correct_values_a = [
            1.0,
            2.0,
            3.0,
            4.0,
            2.0,
            3.0,
            4.0,
            3.0,
            4.0,
            4.0,
            10.0,
            11.0,
            11.0,
        ]
        correct_values_b = [
            5.0,
            6.0,
            7.0,
            8.0,
            6.0,
            7.0,
            8.0,
            7.0,
            8.0,
            8.0,
            12.0,
            13.0,
            13.0,
        ]

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=-1,
            n_jobs=0,
        )

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=-1,
            max_timeshift=None,
            n_jobs=0,
        )

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=-1,
            max_timeshift=1,
            n_jobs=0,
        )

        correct_indices = [
            (1, 0),
            (1, 0),
            (1, 1),
            (1, 1),
            (1, 2),
            (1, 2),
            (1, 3),
            (2, 20),
            (2, 20),
            (2, 21),
        ]
        correct_values_a = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 10.0, 11.0, 11.0]
        correct_values_b = [5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 12.0, 13.0, 13.0]

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=-1,
            max_timeshift=2,
            n_jobs=0,
        )

        correct_indices = [
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 2),
            (1, 2),
            (1, 3),
            (2, 20),
            (2, 20),
            (2, 21),
        ]
        correct_values_a = [
            1.0,
            2.0,
            3.0,
            2.0,
            3.0,
            4.0,
            3.0,
            4.0,
            4.0,
            10.0,
            11.0,
            11.0,
        ]
        correct_values_b = [
            5.0,
            6.0,
            7.0,
            6.0,
            7.0,
            8.0,
            7.0,
            8.0,
            8.0,
            12.0,
            13.0,
            13.0,
        ]

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=-1,
            max_timeshift=4,
            n_jobs=0,
        )

        correct_indices = [
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 2),
            (1, 2),
            (1, 3),
            (2, 20),
            (2, 20),
            (2, 21),
        ]
        correct_values_a = [
            1.0,
            2.0,
            3.0,
            4.0,
            2.0,
            3.0,
            4.0,
            3.0,
            4.0,
            4.0,
            10.0,
            11.0,
            11.0,
        ]
        correct_values_b = [
            5.0,
            6.0,
            7.0,
            8.0,
            6.0,
            7.0,
            8.0,
            7.0,
            8.0,
            8.0,
            12.0,
            13.0,
            13.0,
        ]

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=-1,
            min_timeshift=2,
            max_timeshift=3,
            n_jobs=0,
        )

        correct_indices = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 1), (1, 1), (1, 1)]
        correct_values_a = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0]
        correct_values_b = [5.0, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0]

        self.assertListEqual(list(df["id"].values), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

    def test_rolling_with_larger_shift(self):
        first_class = pd.DataFrame(
            {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "time": range(4)}
        )
        second_class = pd.DataFrame(
            {"a": [10, 11], "b": [12, 13], "time": range(20, 22)}
        )

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
        correct_indices = [
            (1, 1),
            (1, 1),
            (1, 3),
            (1, 3),
            (1, 3),
            (1, 3),
            (2, 21),
            (2, 21),
        ]
        correct_values_a = [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0]
        correct_values_b = [5.0, 6.0, 5.0, 6.0, 7.0, 8.0, 12.0, 13.0]

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=2,
            n_jobs=0,
        )

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        correct_indices = [
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 2),
            (1, 2),
            (2, 20),
            (2, 20),
        ]
        correct_values_a = [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 10.0, 11.0]
        correct_values_b = [5.0, 6.0, 7.0, 8.0, 7.0, 8.0, 12.0, 13.0]

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=-2,
            n_jobs=0,
        )

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

    def test_stacked_rolling(self):
        first_class = pd.DataFrame(
            {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "time": range(4)}
        )
        second_class = pd.DataFrame(
            {"a": [10, 11], "b": [12, 13], "time": range(20, 22)}
        )

        first_class["id"] = 1
        second_class["id"] = 2

        df_full = pd.concat([first_class, second_class], ignore_index=True)

        df_stacked = pd.concat(
            [
                df_full[["time", "id", "a"]].rename(columns={"a": "_value"}),
                df_full[["time", "id", "b"]].rename(columns={"b": "_value"}),
            ],
            ignore_index=True,
        )
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

        df = dataframe_functions.roll_time_series(
            df_stacked,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            rolling_direction=-1,
            n_jobs=0,
        )

        correct_indices = (
            [(1, 0)] * 2 * 4
            + [(1, 1)] * 2 * 3
            + [(1, 2)] * 2 * 2
            + [(1, 3)] * 2 * 1
            + [(2, 20)] * 2 * 2
            + [(2, 21)] * 2 * 1
        )
        self.assertListEqual(list(df["id"].values), correct_indices)

        self.assertListEqual(list(df["kind"].values), ["a", "b"] * 13)
        self.assertListEqual(
            list(df["_value"].values),
            [
                1.0,
                5.0,
                2.0,
                6.0,
                3.0,
                7.0,
                4.0,
                8.0,
                2.0,
                6.0,
                3.0,
                7.0,
                4.0,
                8.0,
                3.0,
                7.0,
                4.0,
                8.0,
                4.0,
                8.0,
                10.0,
                12.0,
                11.0,
                13.0,
                11.0,
                13.0,
            ],
        )

    def test_dict_rolling(self):
        df_dict = {
            "a": pd.DataFrame(
                {"_value": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]}
            ),
            "b": pd.DataFrame(
                {"_value": [5, 6, 7, 8, 12, 13], "id": [1, 1, 1, 1, 2, 2]}
            ),
        }
        df = dataframe_functions.roll_time_series(
            df_dict,
            column_id="id",
            column_sort=None,
            column_kind=None,
            rolling_direction=-1,
            n_jobs=0,
        )
        """ df is
        {a: _value  id
              1.0   1
              2.0   1
              3.0   1
              4.0   1
             10.0   2
             11.0   2,

         b: _value  id
               5.0   1
               6.0   1
               7.0   1
               8.0   1
              12.0   2
              13.0   2
         }
        """

        correct_indices = [
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 0),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 2),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 0),
            (2, 1),
        ]
        self.assertListEqual(list(df["a"]["id"].values), correct_indices)

        self.assertListEqual(list(df["b"]["id"].values), correct_indices)

        self.assertListEqual(
            list(df["a"]["_value"].values),
            [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 3.0, 4.0, 4.0, 10.0, 11.0, 11.0],
        )
        self.assertListEqual(
            list(df["b"]["_value"].values),
            [5.0, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0, 7.0, 8.0, 8.0, 12.0, 13.0, 13.0],
        )

    def test_dict_rolling_maxshift_1(self):
        df_dict = {
            "a": pd.DataFrame(
                {"_value": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]}
            ),
            "b": pd.DataFrame(
                {"_value": [5, 6, 7, 8, 12, 13], "id": [1, 1, 1, 1, 2, 2]}
            ),
        }
        df = dataframe_functions.roll_time_series(
            df_dict,
            column_id="id",
            column_sort=None,
            column_kind=None,
            rolling_direction=-1,
            max_timeshift=1,
            n_jobs=0,
        )
        """ df is
        {a: _value  id
              1.0   1
              2.0   1
              3.0   1
              4.0   1
             10.0   2
             11.0   2,

         b: _value  id
               5.0   1
               6.0   1
               7.0   1
               8.0   1
              12.0   2
              13.0   2
         }
        """

        correct_indices = [
            (1, 0),
            (1, 0),
            (1, 1),
            (1, 1),
            (1, 2),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 0),
            (2, 1),
        ]

        self.assertListEqual(list(df["a"]["id"].values), correct_indices)
        self.assertListEqual(list(df["b"]["id"].values), correct_indices)

        self.assertListEqual(
            list(df["a"]["_value"].values),
            [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 10.0, 11.0, 11.0],
        )
        self.assertListEqual(
            list(df["b"]["_value"].values),
            [5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 12.0, 13.0, 13.0],
        )

    def test_order_rolling(self):

        first_class = pd.DataFrame({"x": [1, 2, 3, 4], "time": [1, 15, 132, 145]})
        second_class = pd.DataFrame({"x": [5, 6, 7], "time": [16, 133, 146]})

        first_class["initial_id"] = 1
        second_class["initial_id"] = 2
        df_full = pd.concat([first_class, second_class], ignore_index=True)

        # Do not show the warning on non-equidistant time
        with warning_free():
            window_size = 2

            df_rolled = dataframe_functions.roll_time_series(
                df_full,
                column_id="initial_id",
                column_sort="time",
                min_timeshift=window_size - 1,
                max_timeshift=window_size - 1,
            )

        """ df is
        {x: _value  id
              1.0   1
              2.0   1
              3.0   1
              4.0   1
              5.0   2
              6.0   2
              7.0   2,
         }
        """

        correct_indices = [
            (1, 15),
            (1, 15),
            (1, 132),
            (1, 132),
            (1, 145),
            (1, 145),
            (2, 133),
            (2, 133),
            (2, 146),
            (2, 146),
        ]

        self.assertListEqual(list(df_rolled["id"]), correct_indices)

    def test_warning_on_non_uniform_time_steps(self):
        with warnings.catch_warnings(record=True) as w:
            first_class = pd.DataFrame(
                {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "time": [1, 2, 4, 5]}
            )
            second_class = pd.DataFrame(
                {"a": [10, 11], "b": [12, 13], "time": list(range(20, 22))}
            )

            first_class["id"] = 1
            second_class["id"] = 2

            df_full = pd.concat([first_class, second_class], ignore_index=True)

            dataframe_functions.roll_time_series(
                df_full,
                column_id="id",
                column_sort="time",
                column_kind=None,
                rolling_direction=1,
                n_jobs=0,
            )

            self.assertGreaterEqual(len(w), 1)
            self.assertIn(
                "Your time stamps are not uniformly sampled, which makes rolling "
                "nonsensical in some domains.",
                [str(warning.message) for warning in w],
            )

    def test_multicore_rolling(self):
        first_class = pd.DataFrame(
            {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "time": range(4)}
        )
        second_class = pd.DataFrame(
            {"a": [10, 11], "b": [12, 13], "time": range(20, 22)}
        )

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
        correct_indices = [
            (1, 0),
            (1, 1),
            (1, 1),
            (1, 2),
            (1, 2),
            (1, 2),
            (1, 3),
            (1, 3),
            (1, 3),
            (1, 3),
            (2, 20),
            (2, 21),
            (2, 21),
        ]
        correct_values_a = [
            1.0,
            1.0,
            2.0,
            1.0,
            2.0,
            3.0,
            1.0,
            2.0,
            3.0,
            4.0,
            10.0,
            10.0,
            11.0,
        ]
        correct_values_b = [
            5.0,
            5.0,
            6.0,
            5.0,
            6.0,
            7.0,
            5.0,
            6.0,
            7.0,
            8.0,
            12.0,
            12.0,
            13.0,
        ]

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=1,
        )

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)

        df = dataframe_functions.roll_time_series(
            df_full,
            column_id="id",
            column_sort="time",
            column_kind=None,
            rolling_direction=1,
            n_jobs=0,
        )

        self.assertListEqual(list(df["id"]), correct_indices)
        self.assertListEqual(list(df["a"].values), correct_values_a)
        self.assertListEqual(list(df["b"].values), correct_values_b)


class CheckForNanTestCase(TestCase):
    def test_all_columns(self):
        test_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[0, 1])

        # should not raise an exception
        dataframe_functions.check_for_nans_in_columns(test_df)

        test_df = pd.DataFrame([[1, 2, 3], [4, np.NaN, 6]], index=[0, 1])

        self.assertRaises(
            ValueError, dataframe_functions.check_for_nans_in_columns, test_df
        )

    def test_not_all_columns(self):
        test_df = pd.DataFrame(
            [[1, 2, 3], [4, np.NaN, 6]], index=[0, 1], columns=["a", "b", "c"]
        )

        self.assertRaises(
            ValueError, dataframe_functions.check_for_nans_in_columns, test_df
        )
        self.assertRaises(
            ValueError,
            dataframe_functions.check_for_nans_in_columns,
            test_df,
            ["a", "b"],
        )
        self.assertRaises(
            ValueError, dataframe_functions.check_for_nans_in_columns, test_df, ["b"]
        )
        self.assertRaises(
            ValueError, dataframe_functions.check_for_nans_in_columns, test_df, "b"
        )
        self.assertRaises(
            ValueError,
            dataframe_functions.check_for_nans_in_columns,
            test_df,
            ["c", "b"],
        )

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

        df = pd.DataFrame(
            [{"value": np.NINF}, {"value": np.NaN}, {"value": np.PINF}, {"value": 1}]
        )
        dataframe_functions.impute_dataframe_zero(df)
        self.assertEqual(list(df.value), [0, 0, 0, 1])

        df = pd.DataFrame(
            [{"value": np.NINF}, {"value": np.NaN}, {"value": np.PINF}, {"value": 1}]
        )
        df = df.astype(np.float64)
        df = dataframe_functions.impute_dataframe_zero(df)
        self.assertEqual(list(df.value), [0, 0, 0, 1])

        df = pd.DataFrame(
            [{"value": np.NINF}, {"value": np.NaN}, {"value": np.PINF}, {"value": 1}]
        )
        df = df.astype(np.float32)
        df = dataframe_functions.impute_dataframe_zero(df)
        self.assertEqual(list(df.value), [0, 0, 0, 1])

        df = pd.DataFrame([])
        dataframe_functions.impute_dataframe_zero(df)

        self.assertEqual(len(df), 0)

    def test_toplevel_impute(self):
        df = pd.DataFrame(
            np.transpose([[0, 1, 2, np.NaN], [1, np.PINF, 2, 3], [1, -3, np.NINF, 3]]),
            columns=["value_a", "value_b", "value_c"],
        )

        dataframe_functions.impute(df)

        self.assertEqual(list(df.value_a), [0, 1, 2, 1])
        self.assertEqual(list(df.value_b), [1, 3, 2, 3])
        self.assertEqual(list(df.value_c), [1, -3, -3, 3])

        df = pd.DataFrame(
            np.transpose(
                [[0, 1, 2, np.NaN], [1, np.PINF, 2, np.NaN], [np.NaN, -3, np.NINF, 3]]
            ),
            columns=["value_a", "value_b", "value_c"],
        )
        df = df.astype(np.float64)
        dataframe_functions.impute(df)

        self.assertEqual(list(df.value_a), [0, 1, 2, 1])
        self.assertEqual(list(df.value_b), [1, 2, 2, 1.5])
        self.assertEqual(list(df.value_c), [0, -3, -3, 3])

        df = pd.DataFrame(
            np.transpose(
                [[0, 1, 2, np.NaN], [1, np.PINF, 2, 3], [np.PINF, -3, np.NINF, 3]]
            ),
            columns=["value_a", "value_b", "value_c"],
        )
        df = df.astype(np.float32)
        dataframe_functions.impute(df)

        self.assertEqual(list(df.value_a), [0, 1, 2, 1])
        self.assertEqual(list(df.value_b), [1, 3, 2, 3])
        self.assertEqual(list(df.value_c), [3, -3, -3, 3])

        df = pd.DataFrame([])
        dataframe_functions.impute(df)

        self.assertEqual(len(df), 0)

    def test_impute_range(self):
        def get_df():
            return pd.DataFrame(
                np.transpose(
                    [[0, 1, 2, np.NaN], [1, np.PINF, 2, 3], [1, -3, np.NINF, 3]]
                ),
                columns=["value_a", "value_b", "value_c"],
            )

        # check if values are replaced correctly
        df = get_df()
        col_to_max = {"value_a": 200, "value_b": 200, "value_c": 200}
        col_to_min = {"value_a": -134, "value_b": -134, "value_c": -134}
        col_to_median = {"value_a": 55, "value_b": 55, "value_c": 55}
        dataframe_functions.impute_dataframe_range(
            df, col_to_max, col_to_min, col_to_median
        )
        self.assertEqual(list(df.value_a), [0, 1, 2, 55])
        self.assertEqual(list(df.value_b), [1, 200, 2, 3])
        self.assertEqual(list(df.value_c), [1, -3, -134, 3])

        # check for error if column key is missing
        df = get_df()
        col_to_max = {"value_a": 200, "value_b": 200, "value_c": 200}
        col_to_min = {"value_a": -134, "value_b": -134, "value_c": -134}
        col_to_median = {"value_a": 55, "value_c": 55}
        self.assertRaises(
            ValueError,
            dataframe_functions.impute_dataframe_range,
            df,
            col_to_max,
            col_to_min,
            col_to_median,
        )

        # check for no error if column key is too much
        col_to_max = {"value_a": 200, "value_b": 200, "value_c": 200}
        col_to_min = {"value_a": -134, "value_b": -134, "value_c": -134}
        col_to_median = {"value_a": 55, "value_b": 55, "value_c": 55, "value_d": 55}
        dataframe_functions.impute_dataframe_range(
            df, col_to_max, col_to_min, col_to_median
        )

        # check for error if replacement value is not finite
        df = get_df()
        col_to_max = {"value_a": 200, "value_b": np.NaN, "value_c": 200}
        col_to_min = {"value_a": -134, "value_b": -134, "value_c": -134}
        col_to_median = {"value_a": 55, "value_b": 55, "value_c": 55}
        self.assertRaises(
            ValueError,
            dataframe_functions.impute_dataframe_range,
            df,
            col_to_max,
            col_to_min,
            col_to_median,
        )
        df = get_df()
        col_to_max = {"value_a": 200, "value_b": 200, "value_c": 200}
        col_to_min = {"value_a": -134, "value_b": np.NINF, "value_c": -134}
        col_to_median = {"value_a": 55, "value_b": 55, "value_c": 55}
        self.assertRaises(
            ValueError,
            dataframe_functions.impute_dataframe_range,
            df,
            col_to_max,
            col_to_min,
            col_to_median,
        )

        df = get_df()
        col_to_max = {"value_a": 200, "value_b": 200, "value_c": 200}
        col_to_min = {"value_a": -134, "value_b": -134, "value_c": -134}
        col_to_median = {"value_a": 55, "value_b": 55, "value_c": np.PINF}
        self.assertRaises(
            ValueError,
            dataframe_functions.impute_dataframe_range,
            df,
            col_to_max,
            col_to_min,
            col_to_median,
        )

        df = pd.DataFrame([0, 1, 2, 3, 4], columns=["test"])
        col_dict = {"test": 0}
        dataframe_functions.impute_dataframe_range(df, col_dict, col_dict, col_dict)

        self.assertEqual(df.columns, ["test"])
        self.assertListEqual(list(df["test"].values), [0, 1, 2, 3, 4])

        df = pd.DataFrame([])
        dataframe_functions.impute_dataframe_range(df, {}, {}, {})

        self.assertEqual(len(df), 0)


class RestrictTestCase(TestCase):
    def test_restrict_dataframe(self):
        df = pd.DataFrame({"id": [1, 2, 3] * 2})

        df_restricted = dataframe_functions.restrict_input_to_index(df, "id", [2])
        self.assertEqual(list(df_restricted.id), [2, 2])

        df_restricted2 = dataframe_functions.restrict_input_to_index(
            df, "id", [1, 2, 3]
        )
        self.assertTrue(df_restricted2.equals(df))

    def test_restrict_dict(self):
        kind_to_df = {
            "a": pd.DataFrame({"id": [1, 2, 3]}),
            "b": pd.DataFrame({"id": [3, 4, 5]}),
        }

        kind_to_df_restricted = dataframe_functions.restrict_input_to_index(
            kind_to_df, "id", [3]
        )
        self.assertEqual(list(kind_to_df_restricted["a"].id), [3])
        self.assertEqual(list(kind_to_df_restricted["b"].id), [3])

        kind_to_df_restricted2 = dataframe_functions.restrict_input_to_index(
            kind_to_df, "id", [1, 2, 3, 4, 5]
        )
        self.assertTrue(kind_to_df_restricted2["a"].equals(kind_to_df["a"]))
        self.assertTrue(kind_to_df_restricted2["b"].equals(kind_to_df["b"]))

    def test_restrict_wrong(self):
        other_type = np.array([1, 2, 3])

        self.assertRaises(
            TypeError,
            dataframe_functions.restrict_input_to_index,
            other_type,
            "id",
            [1, 2, 3],
        )


class GetRangeValuesPerColumnTestCase(TestCase):
    def test_ignores_non_finite_values(self):
        df = pd.DataFrame([0, 1, 2, 3, np.NaN, np.PINF, np.NINF], columns=["value"])

        (
            col_to_max,
            col_to_min,
            col_to_median,
        ) = dataframe_functions.get_range_values_per_column(df)

        self.assertEqual(col_to_max, {"value": 3})
        self.assertEqual(col_to_min, {"value": 0})
        self.assertEqual(col_to_median, {"value": 1.5})

    def test_range_values_correct_with_even_length(self):
        df = pd.DataFrame([0, 1, 2, 3], columns=["value"])

        (
            col_to_max,
            col_to_min,
            col_to_median,
        ) = dataframe_functions.get_range_values_per_column(df)

        self.assertEqual(col_to_max, {"value": 3})
        self.assertEqual(col_to_min, {"value": 0})
        self.assertEqual(col_to_median, {"value": 1.5})

    def test_range_values_correct_with_uneven_length(self):
        df = pd.DataFrame([0, 1, 2], columns=["value"])

        (
            col_to_max,
            col_to_min,
            col_to_median,
        ) = dataframe_functions.get_range_values_per_column(df)

        self.assertEqual(col_to_max, {"value": 2})
        self.assertEqual(col_to_min, {"value": 0})
        self.assertEqual(col_to_median, {"value": 1})

    def test_no_finite_values_yields_0(self):
        df = pd.DataFrame([np.NaN, np.PINF, np.NINF], columns=["value"])

        with warnings.catch_warnings(record=True) as w:
            (
                col_to_max,
                col_to_min,
                col_to_median,
            ) = dataframe_functions.get_range_values_per_column(df)

            self.assertEqual(len(w), 1)
            self.assertEqual(
                str(w[0].message),
                "The columns ['value'] did not have any finite values. Filling with zeros.",
            )

        self.assertEqual(col_to_max, {"value": 0})
        self.assertEqual(col_to_min, {"value": 0})
        self.assertEqual(col_to_median, {"value": 0})


class MakeForecastingFrameTestCase(TestCase):
    def test_make_forecasting_frame_list(self):
        df, y = dataframe_functions.make_forecasting_frame(
            x=range(4), kind="test", max_timeshift=1, rolling_direction=1
        )
        expected_df = pd.DataFrame(
            {
                "id": [("id", 1), ("id", 2), ("id", 3)],
                "kind": ["test"] * 3,
                "value": [0, 1, 2],
                "time": [0, 1, 2],
            }
        )

        expected_y = pd.Series(
            data=[1, 2, 3], index=[("id", 1), ("id", 2), ("id", 3)], name="value"
        )
        assert_frame_equal(
            df.sort_index(axis=1).reset_index(drop=True), expected_df.sort_index(axis=1)
        )
        assert_series_equal(y, expected_y)

    def test_make_forecasting_frame_range(self):
        df, y = dataframe_functions.make_forecasting_frame(
            x=np.arange(4), kind="test", max_timeshift=1, rolling_direction=1
        )
        expected_df = pd.DataFrame(
            {
                "id": list(zip(["id"] * 3, np.arange(1, 4))),
                "kind": ["test"] * 3,
                "value": np.arange(3),
                "time": [0, 1, 2],
            }
        )
        expected_y = pd.Series(
            data=[1, 2, 3], index=[("id", 1), ("id", 2), ("id", 3)], name="value"
        )
        assert_frame_equal(
            df.sort_index(axis=1).reset_index(drop=True), expected_df.sort_index(axis=1)
        )
        assert_series_equal(y, expected_y)

    def test_make_forecasting_frame_pdSeries(self):

        t_index = pd.date_range("1/1/2011", periods=4, freq="H")
        df, y = dataframe_functions.make_forecasting_frame(
            x=pd.Series(data=range(4), index=t_index),
            kind="test",
            max_timeshift=1,
            rolling_direction=1,
        )

        time_shifts = pd.DatetimeIndex(
            ["2011-01-01 01:00:00", "2011-01-01 02:00:00", "2011-01-01 03:00:00"],
            freq="H",
        )
        expected_y = pd.Series(
            data=[1, 2, 3], index=zip(["id"] * 3, time_shifts), name="value"
        )
        expected_df = pd.DataFrame(
            {
                "id": list(
                    zip(
                        ["id"] * 3,
                        pd.DatetimeIndex(
                            [
                                "2011-01-01 01:00:00",
                                "2011-01-01 02:00:00",
                                "2011-01-01 03:00:00",
                            ]
                        ),
                    )
                ),
                "kind": ["test"] * 3,
                "value": [0, 1, 2],
                "time": pd.DatetimeIndex(
                    [
                        "2011-01-01 00:00:00",
                        "2011-01-01 01:00:00",
                        "2011-01-01 02:00:00",
                    ]
                ),
            }
        )
        assert_frame_equal(
            df.sort_index(axis=1).reset_index(drop=True), expected_df.sort_index(axis=1)
        )
        assert_series_equal(y, expected_y)

    def test_make_forecasting_frame_feature_extraction(self):
        t_index = pd.date_range("1/1/2011", periods=4, freq="H")
        df, y = dataframe_functions.make_forecasting_frame(
            x=pd.Series(data=range(4), index=t_index),
            kind="test",
            max_timeshift=1,
            rolling_direction=1,
        )

        extract_relevant_features(
            df,
            y,
            column_id="id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=MinimalFCParameters(),
        )


class GetIDsTestCase(TestCase):
    def test_get_id__correct_DataFrame(self):
        df = pd.DataFrame({"_value": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]})
        self.assertEqual(dataframe_functions.get_ids(df, "id"), {1, 2})

    def test_get_id__correct_dict(self):
        df_dict = {
            "a": pd.DataFrame(
                {"_value": [1, 2, 3, 4, 10, 11], "id": [1, 1, 1, 1, 2, 2]}
            ),
            "b": pd.DataFrame(
                {"_value": [5, 6, 7, 8, 12, 13], "id": [4, 4, 3, 3, 2, 2]}
            ),
        }
        self.assertEqual(dataframe_functions.get_ids(df_dict, "id"), {1, 2, 3, 4})

    def test_get_id_wrong(self):
        other_type = np.array([1, 2, 3])
        self.assertRaises(TypeError, dataframe_functions.get_ids, other_type, "id")


class AddSubIdTestCase(TestCase):
    def test_no_parameters(self):
        dataframe = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        extended_dataframe = dataframe_functions.add_sub_time_series_index(dataframe, 2)

        self.assertEqual(list(extended_dataframe["id"]), [0, 0, 1, 1, 2, 2, 3, 3, 4])
        assert_series_equal(dataframe["value"], extended_dataframe["value"])

    def test_id_parameters(self):
        dataframe = pd.DataFrame(
            {"value": [1, 2, 3, 4, 5, 6, 7, 8, 9], "id": [1, 1, 1, 1, 2, 2, 2, 2, 2]}
        )

        extended_dataframe = dataframe_functions.add_sub_time_series_index(
            dataframe, 2, column_id="id"
        )

        self.assertEqual(
            list(extended_dataframe["id"]),
            [(0, 1), (0, 1), (1, 1), (1, 1), (0, 2), (0, 2), (1, 2), (1, 2), (2, 2)],
        )
        assert_series_equal(dataframe["value"], extended_dataframe["value"])

    def test_kind_parameters(self):
        dataframe = pd.DataFrame(
            {
                "value": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "id": [1, 1, 1, 1, 2, 2, 2, 2, 2],
                "kind": [0, 1, 0, 1, 0, 1, 0, 1, 0],
            }
        )

        extended_dataframe = dataframe_functions.add_sub_time_series_index(
            dataframe, 2, column_id="id", column_kind="kind"
        )

        self.assertEqual(
            list(extended_dataframe["id"]),
            [(0, 1), (0, 1), (0, 1), (0, 1), (0, 2), (0, 2), (0, 2), (0, 2), (1, 2)],
        )
        assert_series_equal(dataframe["value"], extended_dataframe["value"])
        assert_series_equal(dataframe["kind"], extended_dataframe["kind"])

    def test_sort_parameters(self):
        dataframe = pd.DataFrame(
            {
                "value": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "id": [1, 1, 1, 1, 2, 2, 2, 2, 2],
                "kind": [0, 1, 0, 1, 0, 1, 0, 1, 0],
                "sort": [9, 8, 7, 6, 5, 4, 3, 2, 1],
            }
        )

        extended_dataframe = dataframe_functions.add_sub_time_series_index(
            dataframe, 2, column_id="id", column_kind="kind", column_sort="sort"
        )

        self.assertEqual(
            list(extended_dataframe["id"]),
            [(0, 2), (0, 2), (0, 2), (0, 2), (1, 2), (0, 1), (0, 1), (0, 1), (0, 1)],
        )
        self.assertEqual(list(extended_dataframe["value"]), [9, 8, 7, 6, 5, 4, 3, 2, 1])
        self.assertEqual(list(extended_dataframe["kind"]), [0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(list(extended_dataframe["sort"]), [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_dict_input(self):
        dataframe = pd.DataFrame(
            {"value": [1, 2, 3, 4, 5, 6, 7, 8, 9], "id": [1, 1, 1, 1, 2, 2, 2, 2, 2]}
        )

        extended_dataframe = dataframe_functions.add_sub_time_series_index(
            {"1": dataframe}, 2, column_id="id"
        )

        self.assertIn("1", extended_dataframe)

        extended_dataframe = extended_dataframe["1"]

        self.assertEqual(
            list(extended_dataframe["id"]),
            [(0, 1), (0, 1), (1, 1), (1, 1), (0, 2), (0, 2), (1, 2), (1, 2), (2, 2)],
        )
        assert_series_equal(dataframe["value"], extended_dataframe["value"])
