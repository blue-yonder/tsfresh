# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import os
import tempfile

import numpy as np
import pandas as pd
from mock import Mock

from tests.fixtures import DataTestCase
from tsfresh.feature_extraction.extraction import extract_features
from tsfresh.feature_extraction.settings import (
    ComprehensiveFCParameters,
    PickableSettings,
)
from tsfresh.utilities.distribution import IterableDistributorBaseClass, MapDistributor


class ExtractionTestCase(DataTestCase):
    """The unit tests in this module make sure if the time series features are created properly"""

    def setUp(self):
        self.n_jobs = 1
        self.directory = tempfile.gettempdir()

    def test_extract_features(self):
        # todo: implement more methods and test more aspects
        df = self.create_test_data_sample()
        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
        )
        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(np.all(extracted_features.a__maximum == np.array([71, 77])))
        self.assertTrue(
            np.all(extracted_features.a__sum_values == np.array([691, 1017]))
        )
        self.assertTrue(
            np.all(extracted_features.a__abs_energy == np.array([32211, 63167]))
        )
        self.assertTrue(
            np.all(extracted_features.b__sum_values == np.array([757, 695]))
        )
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(
            np.all(extracted_features.b__abs_energy == np.array([36619, 35483]))
        )
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))

        df_sts = self.create_one_valued_time_series()
        extracted_features_sts = extract_features(
            df_sts,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
        )

        self.assertIsInstance(extracted_features_sts, pd.DataFrame)
        self.assertTrue(
            np.all(extracted_features_sts.a__maximum == np.array([1.0, 6.0]))
        )
        self.assertTrue(
            np.all(extracted_features_sts.a__sum_values == np.array([1.0, 11.0]))
        )
        self.assertTrue(
            np.all(extracted_features_sts.a__count_above_mean == np.array([0, 1]))
        )

    def test_extract_features_uses_only_kind_to_fc_settings(self):
        df = self.create_test_data_sample()
        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
            kind_to_fc_parameters={"a": {"maximum": None, "minimum": None}},
        )
        assert len(extracted_features) == 2

    def test_extract_features_for_one_time_series(self):
        # todo: implement more methods and test more aspects
        df = self.create_test_data_sample()
        settings = ComprehensiveFCParameters()

        extracted_features = extract_features(
            df,
            default_fc_parameters=settings,
            column_value="val",
            column_id="id",
            column_kind="kind",
            column_sort="sort",
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(
            np.all(extracted_features.b__sum_values == np.array([757, 695]))
        )
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(
            np.all(extracted_features.b__abs_energy == np.array([36619, 35483]))
        )
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))

        df_sts = self.create_one_valued_time_series()
        extracted_features_sts = extract_features(
            df_sts,
            default_fc_parameters=settings,
            column_value="val",
            column_id="id",
            column_kind="kind",
            column_sort="sort",
        )

        self.assertIsInstance(extracted_features_sts, pd.DataFrame)
        self.assertTrue(
            np.all(extracted_features_sts.a__maximum == np.array([1.0, 6.0]))
        )
        self.assertTrue(
            np.all(extracted_features_sts.a__sum_values == np.array([1.0, 11.0]))
        )
        self.assertTrue(
            np.all(extracted_features_sts.a__count_above_mean == np.array([0, 1]))
        )

    def test_extract_features_for_index_based_functions(self):
        df = self.create_test_data_sample_with_time_index()

        settings = {
            "linear_trend_timewise": [{"attr": "slope"}],
            "linear_trend": [{"attr": "slope"}],
        }

        extracted_features = extract_features(
            df,
            default_fc_parameters=settings,
            column_value="val",
            column_id="id",
            column_kind="kind",
            column_sort="sort",
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)

        slope_a = extracted_features['a__linear_trend_timewise__attr_"slope"'].values
        slope_b = extracted_features['b__linear_trend_timewise__attr_"slope"'].values

        self.assertAlmostEqual(slope_a[0], -0.001347117)
        self.assertAlmostEqual(slope_a[1], 0.052036340)
        self.assertAlmostEqual(slope_b[0], 0.021898496)
        self.assertAlmostEqual(slope_b[1], -0.012312)

        # Test that the index of the returned df is the ID and not the timestamp
        self.assertTrue(extracted_features.index.dtype != df.index.dtype)
        self.assertTrue(extracted_features.index.dtype == df["id"].dtype)
        self.assertEqual(
            sorted(extracted_features.index.unique().tolist()),
            sorted(df["id"].unique().tolist()),
        )

    def test_extract_features_custom_function(self):
        df = self.create_test_data_sample()

        def custom_function(x, p):
            return len(x) + p

        settings = PickableSettings(
            {"mean": None, custom_function: [{"p": 1}, {"p": -1}],}
        )

        extracted_features = extract_features(
            df,
            default_fc_parameters=settings,
            column_value="val",
            column_id="id",
            column_kind="kind",
            column_sort="sort",
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)

        mean_a = extracted_features["a__mean"].values
        custom_function_a_1 = extracted_features["a__custom_function__p_1"].values
        custom_function_a_m1 = extracted_features["a__custom_function__p_-1"].values

        self.assertAlmostEqual(mean_a[0], 34.55)
        self.assertAlmostEqual(mean_a[1], 50.85)
        self.assertAlmostEqual(custom_function_a_1[0], 21)
        self.assertAlmostEqual(custom_function_a_1[1], 21)
        self.assertAlmostEqual(custom_function_a_m1[0], 19)
        self.assertAlmostEqual(custom_function_a_m1[1], 19)

    def test_extract_features_after_randomisation(self):
        df = self.create_test_data_sample()
        df_random = df.copy().sample(frac=1)

        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
        ).sort_index()
        extracted_features_from_random = extract_features(
            df_random,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
        ).sort_index()

        self.assertCountEqual(
            extracted_features.columns, extracted_features_from_random.columns
        )

        for col in extracted_features:
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    extracted_features[col], extracted_features_from_random[col]
                )
            )

    def test_profiling_file_written_out(self):

        df = pd.DataFrame(
            data={"id": np.repeat([1, 2], 10), "val": np.random.normal(0, 1, 20)}
        )
        profiling_filename = os.path.join(self.directory, "test_profiling.txt")
        X = extract_features(
            df,
            column_id="id",
            column_value="val",
            n_jobs=self.n_jobs,
            profile=True,
            profiling_filename=profiling_filename,
        )

        self.assertTrue(os.path.isfile(profiling_filename))
        os.remove(profiling_filename)

    def test_profiling_cumulative_file_written_out(self):

        PROFILING_FILENAME = os.path.join(
            self.directory, "test_profiling_cumulative.txt"
        )
        PROFILING_SORTING = "cumulative"

        df = pd.DataFrame(
            data={"id": np.repeat([1, 2], 10), "val": np.random.normal(0, 1, 20)}
        )
        extract_features(
            df,
            column_id="id",
            column_value="val",
            n_jobs=self.n_jobs,
            profile=True,
            profiling_filename=PROFILING_FILENAME,
            profiling_sorting=PROFILING_SORTING,
        )

        self.assertTrue(os.path.isfile(PROFILING_FILENAME))
        os.remove(PROFILING_FILENAME)

    def test_extract_features_without_settings(self):
        df = pd.DataFrame(
            data={
                "id": np.repeat([1, 2], 10),
                "value1": np.random.normal(0, 1, 20),
                "value2": np.random.normal(0, 1, 20),
            }
        )
        X = extract_features(df, column_id="id", n_jobs=self.n_jobs)
        self.assertIn("value1__maximum", list(X.columns))
        self.assertIn("value2__maximum", list(X.columns))

    def test_extract_features_with_and_without_parallelization(self):
        df = self.create_test_data_sample()

        features_parallel = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=2,
        )

        features_serial = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=0,
        )

        self.assertCountEqual(features_parallel.columns, features_serial.columns)

        for col in features_parallel.columns:
            np.testing.assert_array_almost_equal(
                features_parallel[col], features_serial[col]
            )

    def test_extract_index_preservation(self):
        df = self.create_test_data_nearly_numerical_indices()
        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertEqual(set(df["id"]), set(extracted_features.index))

    def test_extract_features_alphabetically_sorted(self):
        df = self.create_test_data_sample()

        features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
        )

        for col_name in features.columns:
            # split out the configuration of the features calculator
            col_name_chunks = col_name.split("__")
            # the name is always at the beginning, so remove it. Also remove the kind of the column
            col_name_chunks = col_name_chunks[2:]

            self.assertEqual(col_name_chunks, list(sorted(col_name_chunks)))


class ParallelExtractionTestCase(DataTestCase):
    def setUp(self):
        self.n_jobs = 2

        # only calculate some features to reduce load on travis ci
        self.name_to_param = {
            "maximum": None,
            "sum_values": None,
            "abs_energy": None,
            "minimum": None,
            "mean": None,
            "median": None,
        }

    def test_extract_features(self):
        # todo: implement more methods and test more aspects
        df = self.create_test_data_sample()
        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(np.all(extracted_features.a__maximum == np.array([71, 77])))
        self.assertTrue(
            np.all(extracted_features.a__sum_values == np.array([691, 1017]))
        )
        self.assertTrue(
            np.all(extracted_features.a__abs_energy == np.array([32211, 63167]))
        )
        self.assertTrue(
            np.all(extracted_features.b__sum_values == np.array([757, 695]))
        )
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(
            np.all(extracted_features.b__abs_energy == np.array([36619, 35483]))
        )
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))


class DistributorUsageTestCase(DataTestCase):
    def setUp(self):
        # only calculate some features to reduce load on travis ci
        self.name_to_param = {"maximum": None}

    def test_distributor_map_reduce_is_called(self):
        df = self.create_test_data_sample()

        mock = Mock(spec=IterableDistributorBaseClass)
        mock.close.return_value = None
        mock.map_reduce.return_value = []

        X = extract_features(
            timeseries_container=df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            default_fc_parameters=self.name_to_param,
            distributor=mock,
        )

        self.assertTrue(mock.map_reduce.called)

    def test_distributor_close_is_called(self):
        df = self.create_test_data_sample()

        mock = MapDistributor()
        mock.close = Mock()
        mock.close.return_value = None

        X = extract_features(
            timeseries_container=df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            default_fc_parameters=self.name_to_param,
            distributor=mock,
        )

        self.assertTrue(mock.close.called)
