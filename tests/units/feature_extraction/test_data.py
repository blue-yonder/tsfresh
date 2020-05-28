import pandas as pd

from tests.fixtures import DataTestCase
from tsfresh.feature_extraction.data import to_tsdata, LongTsFrameAdapter, WideTsFrameAdapter, TsDictAdapter


class DataAdapterTestCase(DataTestCase):

    def test_long_tsframe(self):
        df = self.create_test_data_sample()
        data = LongTsFrameAdapter(df, "id", "kind", "val", "sort")

        self.assert_tsdata(data, size=4)

    def test_wide_tsframe(self):
        df = self.create_test_data_sample_wide()
        data = WideTsFrameAdapter(df, "id", "sort")

        self.assert_tsdata(data, size=4)

    def test_dict_tsframe(self):
        df = {key: df for key, df in self.create_test_data_sample().groupby(["kind"])}
        data = TsDictAdapter(df, "id", "val", "sort")

        self.assert_tsdata(data, size=4)

    def assert_tsdata(self, data, size):
        self.assertEqual(len(data), size)
        self.assertEqual(sum(1 for _ in data), len(data))
        self.assertEqual(sum(1 for _ in data.partition(1)), size)
        self.assertEqual((sum(sum(1 for _ in g) for g in data.partition(1))), len(data))

    def assert_data_chunk_object_equal(self, result, expected):
        dic_result = {str(x[0]) + "_" + str(x[1]): x[2] for x in result}
        dic_expected = {str(x[0]) + "_" + str(x[1]): x[2] for x in expected}
        for k in dic_result.keys():
            pd.testing.assert_series_equal(dic_result[k], dic_expected[k])

    def test_simple_data_sample_two_timeseries(self):
        df = pd.DataFrame({"id": [10] * 4, "kind": ["a"] * 2 + ["b"] * 2, "val": [36, 71, 78, 37]})
        df.set_index("id", drop=False, inplace=True)
        df.index.name = None

        result = to_tsdata(df, "id", "kind", "val")
        expected = [(10, 'a', pd.Series([36, 71], index=[10] * 2, name="val")),
                    (10, 'b', pd.Series([78, 37], index=[10] * 2, name="val"))]
        self.assert_data_chunk_object_equal(result, expected)

    def test_simple_data_sample_four_timeseries(self):
        df = self.create_test_data_sample()
        # todo: investigate the names that are given
        df.index.name = None
        df.sort_values(by=["id", "kind", "sort"], inplace=True)

        result = to_tsdata(df, "id", "kind", "val", "sort")
        expected = [(10, 'a', pd.Series([36, 71, 27, 62, 56, 58, 67, 11, 2, 24, 45, 30, 0,
                                         9, 41, 28, 33, 19, 29, 43],
                                        index=[10] * 20, name="val")),
                    (10, 'b', pd.Series([78, 37, 23, 44, 6, 3, 21, 61, 39, 31, 53, 16, 66,
                                         50, 40, 47, 7, 42, 38, 55],
                                        index=[10] * 20, name="val")),
                    (500, 'a', pd.Series([76, 72, 74, 75, 32, 64, 46, 35, 15, 70, 57, 65,
                                          51, 26, 5, 25, 10, 69, 73, 77],
                                         index=[500] * 20, name="val")),
                    (500, 'b', pd.Series([8, 60, 12, 68, 22, 17, 18, 63, 49, 34, 20, 52,
                                          48, 14, 79, 4, 1, 59, 54, 13],
                                         index=[500] * 20, name="val"))]

        self.assert_data_chunk_object_equal(result, expected)

    def test_with_dictionaries_two_rows(self):
        test_df = pd.DataFrame([{"value": 2, "sort": 2, "id": "id_1"},
                                {"value": 1, "sort": 1, "id": "id_1"}])
        test_dict = {"a": test_df, "b": test_df}

        result = to_tsdata(test_dict, column_id="id", column_value="value", column_sort="sort")
        expected = [("id_1", 'a', pd.Series([1, 2], index=[1, 0], name="value")),
                    ("id_1", 'b', pd.Series([1, 2], index=[1, 0], name="value"))]
        self.assert_data_chunk_object_equal(result, expected)

    def test_wide_dataframe_order_preserved_with_sort_column(self):
        """ verifies that the order of the sort column from a wide time series container is preserved
        """

        test_df = pd.DataFrame({'id': ["a", "a", "b"],
                                'v1': [3, 2, 1],
                                'v2': [13, 12, 11],
                                'sort': [103, 102, 101]})

        result = to_tsdata(test_df, column_id="id", column_sort="sort")
        expected = [("a", 'v1', pd.Series([2, 3], index=[1, 0], name="v1")),
                    ("a", 'v2', pd.Series([12, 13], index=[1, 0], name="v2")),
                    ("b", 'v1', pd.Series([1], index=[2], name="v1")),
                    ("b", 'v2', pd.Series([11], index=[2], name="v2"))]
        self.assert_data_chunk_object_equal(result, expected)
