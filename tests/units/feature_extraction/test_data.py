import math
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import pandas as pd
from dask import dataframe as dd

from tests.fixtures import DataTestCase
from tsfresh.feature_extraction.data import (
    LongTsFrameAdapter,
    PartitionedTsData,
    TsDictAdapter,
    WideTsFrameAdapter,
    to_tsdata,
)
from tsfresh.utilities.distribution import MultiprocessingDistributor

TEST_DATA_EXPECTED_TUPLES = [
    (
        10,
        "a",
        pd.Series(
            [
                36,
                71,
                27,
                62,
                56,
                58,
                67,
                11,
                2,
                24,
                45,
                30,
                0,
                9,
                41,
                28,
                33,
                19,
                29,
                43,
            ],
            index=[10] * 20,
        ),
    ),
    (
        10,
        "b",
        pd.Series(
            [
                78,
                37,
                23,
                44,
                6,
                3,
                21,
                61,
                39,
                31,
                53,
                16,
                66,
                50,
                40,
                47,
                7,
                42,
                38,
                55,
            ],
            index=[10] * 20,
        ),
    ),
    (
        500,
        "a",
        pd.Series(
            [
                76,
                72,
                74,
                75,
                32,
                64,
                46,
                35,
                15,
                70,
                57,
                65,
                51,
                26,
                5,
                25,
                10,
                69,
                73,
                77,
            ],
            index=[500] * 20,
        ),
    ),
    (
        500,
        "b",
        pd.Series(
            [
                8,
                60,
                12,
                68,
                22,
                17,
                18,
                63,
                49,
                34,
                20,
                52,
                48,
                14,
                79,
                4,
                1,
                59,
                54,
                13,
            ],
            index=[500] * 20,
        ),
    ),
]

WIDE_TEST_DATA_EXPECTED_TUPLES = [
    (
        10,
        "a",
        pd.Series(
            [
                11,
                9,
                67,
                45,
                30,
                58,
                62,
                19,
                56,
                29,
                0,
                27,
                36,
                43,
                33,
                2,
                24,
                71,
                41,
                28,
            ],
            index=list(range(20)),
        ),
    ),
    (
        10,
        "b",
        pd.Series(
            [
                50,
                40,
                39,
                7,
                53,
                23,
                16,
                37,
                66,
                38,
                6,
                47,
                3,
                61,
                44,
                42,
                78,
                31,
                21,
                55,
            ],
            index=list(range(20)),
        ),
    ),
    (
        500,
        "a",
        pd.Series(
            [
                15,
                35,
                25,
                32,
                69,
                65,
                70,
                64,
                51,
                46,
                5,
                77,
                26,
                73,
                76,
                75,
                72,
                74,
                10,
                57,
            ],
            index=list(range(20, 40)),
        ),
    ),
    (
        500,
        "b",
        pd.Series(
            [
                4,
                14,
                68,
                22,
                18,
                52,
                54,
                60,
                79,
                12,
                49,
                63,
                8,
                59,
                1,
                13,
                20,
                17,
                48,
                34,
            ],
            index=list(range(20, 40)),
        ),
    ),
]


class DataAdapterTestCase(DataTestCase):
    def test_long_tsframe(self):
        df = self.create_test_data_sample()
        data = LongTsFrameAdapter(df, "id", "kind", "val", "sort")

        self.assert_tsdata(data, TEST_DATA_EXPECTED_TUPLES)

    def test_long_tsframe_no_value_column(self):
        df = self.create_test_data_sample()
        data = LongTsFrameAdapter(df, "id", "kind", None, "sort")

        self.assert_tsdata(data, TEST_DATA_EXPECTED_TUPLES)

    def test_wide_tsframe(self):
        df = self.create_test_data_sample_wide()
        data = WideTsFrameAdapter(df, "id", "sort")

        self.assert_tsdata(data, WIDE_TEST_DATA_EXPECTED_TUPLES)

    def test_wide_tsframe_without_sort(self):
        df = self.create_test_data_sample_wide()
        del df["sort"]
        data = WideTsFrameAdapter(df, "id")

        self.assert_tsdata(data, WIDE_TEST_DATA_EXPECTED_TUPLES)

    def test_dict_tsframe(self):
        df = {key: df for key, df in self.create_test_data_sample().groupby(["kind"])}
        data = TsDictAdapter(df, "id", "val", "sort")

        self.assert_tsdata(data, TEST_DATA_EXPECTED_TUPLES)

    def assert_tsdata(self, data, expected):
        self.assertEqual(len(data), len(expected))
        self.assertEqual(sum(1 for _ in data), len(data))
        self.assert_data_chunk_object_equal(data, expected)

    def assert_data_chunk_object_equal(self, result, expected):
        dic_result = {str(x[0]) + "_" + str(x[1]): x[2] for x in result}
        dic_expected = {str(x[0]) + "_" + str(x[1]): x[2] for x in expected}
        for k in dic_result.keys():
            pd.testing.assert_series_equal(
                dic_result[k], dic_expected[k], check_names=False
            )

    def test_simple_data_sample_two_timeseries(self):
        df = pd.DataFrame(
            {"id": [10] * 4, "kind": ["a"] * 2 + ["b"] * 2, "val": [36, 71, 78, 37]}
        )
        df.set_index("id", drop=False, inplace=True)
        df.index.name = None

        result = to_tsdata(df, "id", "kind", "val")
        expected = [
            (10, "a", pd.Series([36, 71], index=[10] * 2, name="val")),
            (10, "b", pd.Series([78, 37], index=[10] * 2, name="val")),
        ]
        self.assert_data_chunk_object_equal(result, expected)

    def test_simple_data_sample_four_timeseries(self):
        df = self.create_test_data_sample()
        df.index.name = None
        df.sort_values(by=["id", "kind", "sort"], inplace=True)

        result = to_tsdata(df, "id", "kind", "val", "sort")
        expected = TEST_DATA_EXPECTED_TUPLES

        self.assert_data_chunk_object_equal(result, expected)

    def test_with_dictionaries_two_rows(self):
        test_df = pd.DataFrame(
            [
                {"value": 2, "sort": 2, "id": "id_1"},
                {"value": 1, "sort": 1, "id": "id_1"},
            ]
        )
        test_dict = {"a": test_df, "b": test_df}

        result = to_tsdata(
            test_dict, column_id="id", column_value="value", column_sort="sort"
        )
        expected = [
            ("id_1", "a", pd.Series([1, 2], index=[1, 0], name="value")),
            ("id_1", "b", pd.Series([1, 2], index=[1, 0], name="value")),
        ]
        self.assert_data_chunk_object_equal(result, expected)

    def test_with_dictionaries_two_rows(self):
        test_df = pd.DataFrame([{"value": 1, "id": "id_1"}, {"value": 2, "id": "id_1"}])
        test_dict = {"a": test_df, "b": test_df}

        result = to_tsdata(test_dict, column_id="id", column_value="value")
        expected = [
            ("id_1", "a", pd.Series([1, 2], index=[0, 1], name="value")),
            ("id_1", "b", pd.Series([1, 2], index=[0, 1], name="value")),
        ]
        self.assert_data_chunk_object_equal(result, expected)

    def test_wide_dataframe_order_preserved_with_sort_column(self):
        """verifies that the order of the sort column from a wide time series container is preserved"""

        test_df = pd.DataFrame(
            {
                "id": ["a", "a", "b"],
                "v1": [3, 2, 1],
                "v2": [13, 12, 11],
                "sort": [103, 102, 101],
            }
        )

        result = to_tsdata(test_df, column_id="id", column_sort="sort")
        expected = [
            ("a", "v1", pd.Series([2, 3], index=[1, 0], name="v1")),
            ("a", "v2", pd.Series([12, 13], index=[1, 0], name="v2")),
            ("b", "v1", pd.Series([1], index=[2], name="v1")),
            ("b", "v2", pd.Series([11], index=[2], name="v2")),
        ]
        self.assert_data_chunk_object_equal(result, expected)

    def test_dask_dataframe_with_kind(self):
        test_df = dd.from_pandas(
            pd.DataFrame({"id": [1, 2], "kind": ["a", "a"], "value": [1, 2]}),
            npartitions=1,
        )

        result = to_tsdata(test_df, column_id="id", column_kind="kind")
        self.assertEqual(result.column_id, "id")
        self.assertEqual(result.column_kind, "kind")
        self.assertEqual(result.column_value, "value")

        def test_f(chunk):
            return pd.DataFrame(
                {"id": chunk[0], "variable": chunk[1], "value": chunk[2]}
            )

        return_f = result.apply(
            test_f, meta=(("id", "int"), ("variable", "int"), ("value", "int"))
        ).compute()
        pd.testing.assert_frame_equal(
            return_f,
            pd.DataFrame({"id": [1, 2], "variable": ["a", "a"], "value": [1.0, 2.0]}),
        )

    def test_dask_dataframe_without_kind(self):
        test_df = dd.from_pandas(
            pd.DataFrame({"id": [1, 2], "value_a": [1, 2], "value_b": [3, 4]}),
            npartitions=1,
        )

        result = to_tsdata(test_df, column_id="id")
        self.assertEqual(result.column_id, "id")

        def test_f(chunk):
            return pd.DataFrame(
                {"id": chunk[0], "variable": chunk[1], "value": chunk[2]}
            )

        return_f = result.apply(
            test_f, meta=(("id", "int"), ("variable", "int"), ("value", "int"))
        ).compute()
        pd.testing.assert_frame_equal(
            return_f.reset_index(drop=True),
            pd.DataFrame(
                {
                    "id": [1, 2, 1, 2],
                    "variable": ["value_a", "value_a", "value_b", "value_b"],
                    "value": [1.0, 2.0, 3.0, 4.0],
                }
            ),
        )

        test_df = dd.from_pandas(
            pd.DataFrame(
                {"id": [1, 1], "sort": [2, 1], "value_a": [1, 2], "value_b": [3, 4]}
            ),
            npartitions=1,
        )

        result = to_tsdata(test_df, column_id="id", column_sort="sort")
        self.assertEqual(result.column_id, "id")

        def test_f(chunk):
            return pd.DataFrame(
                {"id": chunk[0], "variable": chunk[1], "value": chunk[2]}
            )

        return_f = result.apply(
            test_f, meta=(("id", "int"), ("variable", "int"), ("value", "int"))
        ).compute()

        pd.testing.assert_frame_equal(
            return_f.reset_index(drop=True),
            pd.DataFrame(
                {
                    "id": [1, 1, 1, 1],
                    "variable": ["value_a", "value_a", "value_b", "value_b"],
                    "value": [2.0, 1.0, 4.0, 3.0],
                }
            ),
        )

    def test_with_wrong_input(self):
        test_df = pd.DataFrame([{"id": 0, "kind": "a", "value": 3, "sort": np.NaN}])
        self.assertRaises(ValueError, to_tsdata, test_df, "id", "kind", "value", "sort")

        test_df = pd.DataFrame([{"id": 0, "kind": "a", "value": 3, "sort": 1}])
        self.assertRaises(
            ValueError, to_tsdata, test_df, "strange_id", "kind", "value", "sort"
        )
        test_df = dd.from_pandas(test_df, npartitions=1)
        self.assertRaises(
            ValueError, to_tsdata, test_df, "strange_id", "kind", "value", "sort"
        )

        test_df = pd.DataFrame(
            [{"id": 0, "kind": "a", "value": 3, "value_2": 1, "sort": 1}]
        )
        self.assertRaises(
            ValueError, to_tsdata, test_df, "strange_id", "kind", None, "sort"
        )
        test_df = dd.from_pandas(test_df, npartitions=1)
        self.assertRaises(
            ValueError, to_tsdata, test_df, "strange_id", "kind", None, "sort"
        )

        test_df = pd.DataFrame([{"id": 0, "kind": "a", "value": 3, "sort": 1}])
        self.assertRaises(
            ValueError, to_tsdata, test_df, "id", "strange_kind", "value", "sort"
        )
        test_df = dd.from_pandas(test_df, npartitions=1)
        self.assertRaises(
            ValueError, to_tsdata, test_df, "id", "strange_kind", "value", "sort"
        )

        test_df = pd.DataFrame([{"id": np.NaN, "kind": "a", "value": 3, "sort": 1}])
        self.assertRaises(ValueError, to_tsdata, test_df, "id", "kind", "value", "sort")

        test_df = pd.DataFrame([{"id": 0, "kind": np.NaN, "value": 3, "sort": 1}])
        self.assertRaises(ValueError, to_tsdata, test_df, "id", "kind", "value", "sort")

        test_df = pd.DataFrame([{"id": 2}, {"id": 1}])
        test_dd = dd.from_pandas(test_df, npartitions=1)
        test_dict = {"a": test_df, "b": test_df}

        # column_id needs to be given
        self.assertRaises(ValueError, to_tsdata, test_df, None, "a", "b", None)
        self.assertRaises(ValueError, to_tsdata, test_dd, None, "a", "b", None)
        self.assertRaises(ValueError, to_tsdata, test_df, None, "a", "b", "a")
        self.assertRaises(ValueError, to_tsdata, test_dd, None, "a", "b", "a")
        self.assertRaises(ValueError, to_tsdata, test_dict, None, "a", "b", None)
        self.assertRaises(ValueError, to_tsdata, test_dict, None, "a", "b", "a")

        # If there are more than one column, the algorithm can not choose the correct column
        self.assertRaises(ValueError, to_tsdata, test_dict, "id", None, None, None)

        test_dict = {
            "a": pd.DataFrame([{"id": 2, "value_a": 3}, {"id": 1, "value_a": 4}]),
            "b": pd.DataFrame([{"id": 2}, {"id": 1}]),
        }

        # If there are more than one column, the algorithm can not choose the correct column
        self.assertRaises(ValueError, to_tsdata, test_dict, "id", None, None, None)

        test_df = pd.DataFrame([{"id": 0, "value": np.NaN}])
        self.assertRaises(ValueError, to_tsdata, test_df, "id", None, "value", None)

        test_df = pd.DataFrame([{"id": 0, "value": np.NaN}])
        self.assertRaises(ValueError, to_tsdata, test_df, None, None, "value", None)

        test_df = pd.DataFrame([{"id": 0, "a_": 3, "b": 5, "sort": 1}])
        self.assertRaises(ValueError, to_tsdata, test_df, "id", None, None, "sort")
        test_df = dd.from_pandas(test_df, npartitions=1)
        self.assertRaises(ValueError, to_tsdata, test_df, "id", None, None, "sort")

        test_df = pd.DataFrame([{"id": 0, "a__c": 3, "b": 5, "sort": 1}])
        self.assertRaises(ValueError, to_tsdata, test_df, "id", None, None, "sort")
        test_df = dd.from_pandas(test_df, npartitions=1)
        self.assertRaises(ValueError, to_tsdata, test_df, "id", None, None, "sort")

        test_df = pd.DataFrame([{"id": 0}])
        self.assertRaises(ValueError, to_tsdata, test_df, "id", None, None, None)
        test_df = dd.from_pandas(test_df, npartitions=1)
        self.assertRaises(ValueError, to_tsdata, test_df, "id", None, None, None)

        test_df = pd.DataFrame([{"id": 0, "sort": 0}])
        self.assertRaises(ValueError, to_tsdata, test_df, "id", None, None, "sort")
        test_df = dd.from_pandas(test_df, npartitions=1)
        self.assertRaises(ValueError, to_tsdata, test_df, "id", None, None, "sort")

        test_df = [1, 2, 3]
        self.assertRaises(ValueError, to_tsdata, test_df, "a", "b", "c", "d")


class PivotListTestCase(TestCase):
    def test_empty_list(self):
        mock_ts_data = Mock()
        mock_ts_data.df_id_type = str

        return_df = PartitionedTsData.pivot(mock_ts_data, [])

        self.assertEqual(len(return_df), 0)
        self.assertEqual(len(return_df.index), 0)
        self.assertEqual(len(return_df.columns), 0)

    def test_different_input(self):
        mock_ts_data = Mock()
        mock_ts_data.df_id_type = str

        input_list = [
            ("a", "b", 1),
            ("a", "c", 2),
            ("A", "b", 3),
            ("A", "c", 4),
            ("X", "Y", 5),
        ]
        return_df = PartitionedTsData.pivot(mock_ts_data, input_list)

        self.assertEqual(len(return_df), 3)
        self.assertEqual(set(return_df.index), {"a", "A", "X"})
        self.assertEqual(set(return_df.columns), {"b", "c", "Y"})

        self.assertEqual(return_df.loc["a", "b"], 1)
        self.assertEqual(return_df.loc["a", "c"], 2)
        self.assertEqual(return_df.loc["A", "b"], 3)
        self.assertEqual(return_df.loc["A", "c"], 4)
        self.assertEqual(return_df.loc["X", "Y"], 5)

    def test_long_input(self):
        mock_ts_data = Mock()
        mock_ts_data.df_id_type = str

        input_list = []
        for i in range(100):
            for j in range(100):
                input_list.append((i, j, i * j))

        return_df = PartitionedTsData.pivot(mock_ts_data, input_list)

        self.assertEqual(len(return_df), 100)
        self.assertEqual(len(return_df.columns), 100)
        # every cell should be filled
        self.assertEqual(np.sum(np.sum(np.isnan(return_df))), 0)
        # test if all entries are there
        self.assertEqual(return_df.sum().sum(), 24502500)
