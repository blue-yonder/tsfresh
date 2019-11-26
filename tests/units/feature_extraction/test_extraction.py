# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import os
import tempfile

import numpy as np
import pandas as pd
from mock import Mock

from tests.fixtures import DataTestCase
from tsfresh.feature_extraction.extraction import extract_features, generate_data_chunk_format
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.utilities.distribution import DistributorBaseClass


class ExtractionTestCase(DataTestCase):
    """The unit tests in this module make sure if the time series features are created properly"""

    def setUp(self):
        self.n_jobs = 1
        self.directory = tempfile.gettempdir()

    def test_extract_features(self):
        # todo: implement more methods and test more aspects
        df = self.create_test_data_sample()
        extracted_features = extract_features(df, column_id="id", column_sort="sort",
                                              column_kind="kind", column_value="val",
                                              n_jobs=self.n_jobs)
        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(np.all(extracted_features.a__maximum == np.array([71, 77])))
        self.assertTrue(np.all(extracted_features.a__sum_values == np.array([691, 1017])))
        self.assertTrue(np.all(extracted_features.a__abs_energy == np.array([32211, 63167])))
        self.assertTrue(np.all(extracted_features.b__sum_values == np.array([757, 695])))
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(np.all(extracted_features.b__abs_energy == np.array([36619, 35483])))
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))

        df_sts = self.create_one_valued_time_series()
        extracted_features_sts = extract_features(df_sts, column_id="id", column_sort="sort",
                                                  column_kind="kind", column_value="val",
                                                  n_jobs=self.n_jobs)

        self.assertIsInstance(extracted_features_sts, pd.DataFrame)
        self.assertTrue(np.all(extracted_features_sts.a__maximum == np.array([1.0, 6.0])))
        self.assertTrue(np.all(extracted_features_sts.a__sum_values == np.array([1.0, 11.0])))
        self.assertTrue(np.all(extracted_features_sts.a__count_above_mean == np.array([0, 1])))

    def test_extract_features_uses_only_kind_to_fc_settings(self):
        df = self.create_test_data_sample()
        extracted_features = extract_features(df, column_id="id", column_sort="sort", column_kind="kind",
                                              column_value="val", n_jobs=self.n_jobs,
                                              kind_to_fc_parameters={"a": {"maximum": None, "minimum": None}})
        assert len(extracted_features) == 2

    def test_extract_features_for_one_time_series(self):
        # todo: implement more methods and test more aspects
        df = self.create_test_data_sample()
        settings = ComprehensiveFCParameters()

        extracted_features = extract_features(df, default_fc_parameters=settings,
                                              column_value="val", column_id="id",
                                              column_kind="kind", column_sort="sort")

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(np.all(extracted_features.b__sum_values == np.array([757, 695])))
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(np.all(extracted_features.b__abs_energy == np.array([36619, 35483])))
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))

        df_sts = self.create_one_valued_time_series()
        extracted_features_sts = extract_features(df_sts, default_fc_parameters=settings,
                                                  column_value="val", column_id="id",
                                                  column_kind="kind", column_sort="sort")

        self.assertIsInstance(extracted_features_sts, pd.DataFrame)
        self.assertTrue(np.all(extracted_features_sts.a__maximum == np.array([1.0, 6.0])))
        self.assertTrue(np.all(extracted_features_sts.a__sum_values == np.array([1.0, 11.0])))
        self.assertTrue(np.all(extracted_features_sts.a__count_above_mean == np.array([0, 1])))

    def test_extract_features_for_index_based_functions(self):
        df = self.create_test_data_sample_with_time_index()

        settings = {
            'linear_trend_timewise': [{"attr": "slope"}],
            'linear_trend': [{"attr": "slope"}]
        }

        extracted_features = extract_features(df, default_fc_parameters=settings,
                                              column_value="val", column_id="id",
                                              column_kind="kind",
                                              column_sort="sort")

        self.assertIsInstance(extracted_features, pd.DataFrame)

        slope_a = extracted_features['a__linear_trend_timewise__attr_"slope"'].values
        slope_b = extracted_features['b__linear_trend_timewise__attr_"slope"'].values

        self.assertAlmostEqual(slope_a[0], -0.001347117)
        self.assertAlmostEqual(slope_a[1], 0.052036340)
        self.assertAlmostEqual(slope_b[0], 0.021898496)
        self.assertAlmostEqual(slope_b[1], -0.012312)

        # Test that the index of the returned df is the ID and not the timestamp
        self.assertTrue(extracted_features.index.dtype != df.index.dtype)
        self.assertTrue(extracted_features.index.dtype == df['id'].dtype)
        self.assertEqual(
            sorted(extracted_features.index.unique().tolist()), sorted(df['id'].unique().tolist())
        )

    def test_extract_features_after_randomisation(self):
        df = self.create_test_data_sample()
        df_random = df.copy().sample(frac=1)

        extracted_features = extract_features(df, column_id="id", column_sort="sort",
                                              column_kind="kind",
                                              column_value="val",
                                              n_jobs=self.n_jobs).sort_index()
        extracted_features_from_random = extract_features(df_random, column_id="id",
                                                          column_sort="sort",
                                                          column_kind="kind",
                                                          column_value="val",
                                                          n_jobs=self.n_jobs).sort_index()

        self.assertCountEqual(extracted_features.columns,
                              extracted_features_from_random.columns)

        for col in extracted_features:
            self.assertIsNone(np.testing.assert_array_almost_equal(extracted_features[col],
                                                                   extracted_features_from_random[
                                                                       col]))

    def test_profiling_file_written_out(self):

        df = pd.DataFrame(data={"id": np.repeat([1, 2], 10), "val": np.random.normal(0, 1, 20)})
        profiling_filename = os.path.join(self.directory, "test_profiling.txt")
        X = extract_features(df, column_id="id", column_value="val", n_jobs=self.n_jobs,
                             profile=True, profiling_filename=profiling_filename)

        self.assertTrue(os.path.isfile(profiling_filename))
        os.remove(profiling_filename)

    def test_profiling_cumulative_file_written_out(self):

        PROFILING_FILENAME = os.path.join(self.directory, "test_profiling_cumulative.txt")
        PROFILING_SORTING = "cumulative"

        df = pd.DataFrame(data={"id": np.repeat([1, 2], 10), "val": np.random.normal(0, 1, 20)})
        extract_features(df, column_id="id", column_value="val", n_jobs=self.n_jobs,
                         profile=True, profiling_filename=PROFILING_FILENAME,
                         profiling_sorting=PROFILING_SORTING)

        self.assertTrue(os.path.isfile(PROFILING_FILENAME))
        os.remove(PROFILING_FILENAME)

    def test_extract_features_without_settings(self):
        df = pd.DataFrame(data={"id": np.repeat([1, 2], 10),
                                "value1": np.random.normal(0, 1, 20),
                                "value2": np.random.normal(0, 1, 20)})
        X = extract_features(df, column_id="id",
                             n_jobs=self.n_jobs)
        self.assertIn("value1__maximum", list(X.columns))
        self.assertIn("value2__maximum", list(X.columns))

    def test_extract_features_with_and_without_parallelization(self):
        df = self.create_test_data_sample()

        features_parallel = extract_features(df, column_id="id", column_sort="sort",
                                             column_kind="kind", column_value="val",
                                             n_jobs=self.n_jobs)

        features_serial = extract_features(df, column_id="id", column_sort="sort",
                                           column_kind="kind", column_value="val",
                                           n_jobs=0)

        self.assertCountEqual(features_parallel.columns, features_serial.columns)

        for col in features_parallel.columns:
            np.testing.assert_array_almost_equal(features_parallel[col], features_serial[col])

    def test_extract_index_preservation(self):
        df = self.create_test_data_nearly_numerical_indices()
        extracted_features = extract_features(df, column_id="id", column_sort="sort",
                                              column_kind="kind", column_value="val",
                                              n_jobs=self.n_jobs)

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertEqual(set(df["id"]), set(extracted_features.index))


class ParallelExtractionTestCase(DataTestCase):
    def setUp(self):
        self.n_jobs = 2

        # only calculate some features to reduce load on travis ci
        self.name_to_param = {"maximum": None,
                              "sum_values": None,
                              "abs_energy": None,
                              "minimum": None,
                              "mean": None,
                              "median": None}

    def test_extract_features(self):
        # todo: implement more methods and test more aspects
        df = self.create_test_data_sample()
        extracted_features = extract_features(df, column_id="id", column_sort="sort",
                                              column_kind="kind",
                                              column_value="val",
                                              n_jobs=self.n_jobs)

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(np.all(extracted_features.a__maximum == np.array([71, 77])))
        self.assertTrue(np.all(extracted_features.a__sum_values == np.array([691, 1017])))
        self.assertTrue(np.all(extracted_features.a__abs_energy == np.array([32211, 63167])))
        self.assertTrue(np.all(extracted_features.b__sum_values == np.array([757, 695])))
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(np.all(extracted_features.b__abs_energy == np.array([36619, 35483])))
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))


class DistributorUsageTestCase(DataTestCase):
    def setUp(self):
        # only calculate some features to reduce load on travis ci
        self.name_to_param = {"maximum": None}

    def test_assert_is_distributor(self):
        df = self.create_test_data_sample()

        self.assertRaises(ValueError, extract_features,
                          timeseries_container=df, column_id="id", column_sort="sort",
                          column_kind="kind", column_value="val",
                          default_fc_parameters=self.name_to_param, distributor=object())

        self.assertRaises(ValueError, extract_features,
                          timeseries_container=df, column_id="id", column_sort="sort",
                          column_kind="kind", column_value="val",
                          default_fc_parameters=self.name_to_param, distributor=13)

    def test_distributor_map_reduce_and_close_are_called(self):
        df = self.create_test_data_sample()

        mock = Mock(spec=DistributorBaseClass)
        mock.close.return_value = None
        mock.map_reduce.return_value = []

        X = extract_features(timeseries_container=df, column_id="id", column_sort="sort",
                             column_kind="kind", column_value="val",
                             default_fc_parameters=self.name_to_param, distributor=mock)

        self.assertTrue(mock.close.called)
        self.assertTrue(mock.map_reduce.called)


class GenerateDataChunkTestCase(DataTestCase):

    def assert_data_chunk_object_equal(self, result, expected):
        dic_result = {str(x[0]) + "_" + str(x[1]): x[2] for x in result}
        dic_expected = {str(x[0]) + "_" + str(x[1]): x[2] for x in expected}
        for k in dic_result.keys():
            pd.testing.assert_series_equal(dic_result[k], dic_expected[k])

    def test_simple_data_sample_two_timeseries(self):
        df = pd.DataFrame({"id": [10] * 4, "kind": ["a"] * 2 + ["b"] * 2, "val": [36, 71, 78, 37]})
        df.set_index("id", drop=False, inplace=True)
        df.index.name = None

        result = generate_data_chunk_format(df, "id", "kind", "val")
        expected = [(10, 'a', pd.Series([36, 71], index=[10] * 2, name="val")),
                    (10, 'b', pd.Series([78, 37], index=[10] * 2, name="val"))]
        self.assert_data_chunk_object_equal(result, expected)

    def test_simple_data_sample_four_timeseries(self):
        df = self.create_test_data_sample()
        # todo: investigate the names that are given
        df.index.name = None
        df.sort_values(by=["id", "kind", "sort"], inplace=True)

        result = generate_data_chunk_format(df, "id", "kind", "val")
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
