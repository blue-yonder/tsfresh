# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from __future__ import absolute_import, division

import os

import numpy as np
import pandas as pd
import six
from mock import Mock

from tests.fixtures import DataTestCase
from tsfresh.feature_extraction.extraction import extract_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters

import tempfile

from tsfresh.utilities.distribution import DistributorBaseClass


class ExtractionTestCase(DataTestCase):
    """The unit tests in this module make sure if the time series features are created properly"""

    def setUp(self):
        self.n_jobs = 1
        self.directory = tempfile.gettempdir()

    def test_extract_features(self):
        # todo: implement more methods and test more aspects
        df = self.create_test_data_sample()
        extracted_features = extract_features(df, column_id="id", column_sort="sort", column_kind="kind",
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

        df_sts = self.create_one_valued_time_series()
        extracted_features_sts = extract_features(df_sts, column_id="id", column_sort="sort", column_kind="kind",
                                                  column_value="val",
                                                  n_jobs=self.n_jobs)

        self.assertIsInstance(extracted_features_sts, pd.DataFrame)
        self.assertTrue(np.all(extracted_features_sts.a__maximum == np.array([1.0, 6.0])))
        self.assertTrue(np.all(extracted_features_sts.a__sum_values == np.array([1.0, 11.0])))
        self.assertTrue(np.all(extracted_features_sts.a__count_above_mean == np.array([0, 1])))

    def test_extract_features_for_one_time_series(self):
        # todo: implement more methods and test more aspects
        df = self.create_test_data_sample()
        settings = ComprehensiveFCParameters()

        extracted_features = extract_features(df, default_fc_parameters=settings,
                                              column_value="val", column_id="id", column_kind="kind",
                                              column_sort="sort")

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(np.all(extracted_features.b__sum_values == np.array([757, 695])))
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(np.all(extracted_features.b__abs_energy == np.array([36619, 35483])))
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))

        df_sts = self.create_one_valued_time_series()
        extracted_features_sts = extract_features(df_sts, default_fc_parameters=settings,
                                                  column_value="val", column_id="id", column_kind="kind",
                                                  column_sort="sort")

        self.assertIsInstance(extracted_features_sts, pd.DataFrame)
        self.assertTrue(np.all(extracted_features_sts.a__maximum == np.array([1.0, 6.0])))
        self.assertTrue(np.all(extracted_features_sts.a__sum_values == np.array([1.0, 11.0])))
        self.assertTrue(np.all(extracted_features_sts.a__count_above_mean == np.array([0, 1])))

    def test_extract_features_after_randomisation(self):
        df = self.create_test_data_sample()
        df_random = df.copy().sample(frac=1)

        extracted_features = extract_features(df, column_id="id", column_sort="sort", column_kind="kind",
                                              column_value="val",
                                              n_jobs=self.n_jobs).sort_index()
        extracted_features_from_random = extract_features(df_random, column_id="id", column_sort="sort",
                                                          column_kind="kind",
                                                          column_value="val",
                                                          n_jobs=self.n_jobs).sort_index()

        six.assertCountEqual(self, extracted_features.columns, extracted_features_from_random.columns)

        for col in extracted_features:
            self.assertIsNone(np.testing.assert_array_almost_equal(extracted_features[col],
                                                                   extracted_features_from_random[col]))

    def test_profiling_file_written_out(self):

        df = pd.DataFrame(data={"id": np.repeat([1, 2], 10), "val": np.random.normal(0, 1, 20)})
        profiling_filename = os.path.join(self.directory, "test_profiling.txt")
        X = extract_features(df, column_id="id",
                             column_value="val", n_jobs=self.n_jobs,
                             profile=True, profiling_filename=profiling_filename)

        self.assertTrue(os.path.isfile(profiling_filename))
        os.remove(profiling_filename)

    def test_profiling_cumulative_file_written_out(self):

        PROFILING_FILENAME = os.path.join(self.directory, "test_profiling_cumulative.txt")
        PROFILING_SORTING = "cumulative"

        df = pd.DataFrame(data={"id": np.repeat([1, 2], 10), "val": np.random.normal(0, 1, 20)})
        extract_features(df, column_id="id",
                         column_value="val", n_jobs=self.n_jobs,
                         profile=True, profiling_filename=PROFILING_FILENAME, profiling_sorting=PROFILING_SORTING)

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

        features_parallel = extract_features(df, column_id="id", column_sort="sort", column_kind="kind",
                                             column_value="val",
                                             n_jobs=self.n_jobs)
        features_serial = extract_features(df, column_id="id", column_sort="sort", column_kind="kind",
                                           column_value="val", n_jobs=0)

        six.assertCountEqual(self, features_parallel.columns, features_serial.columns)

        for col in features_parallel.columns:
            np.testing.assert_array_almost_equal(features_parallel[col], features_serial[col])


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
        extracted_features = extract_features(df, column_id="id", column_sort="sort", column_kind="kind",
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
                          timeseries_container=df, column_id="id", column_sort="sort", column_kind="kind",
                          column_value="val", default_fc_parameters=self.name_to_param,
                          distributor=object())

        self.assertRaises(ValueError, extract_features,
                          timeseries_container=df, column_id="id", column_sort="sort", column_kind="kind",
                          column_value="val", default_fc_parameters=self.name_to_param,
                          distributor=13)

    def test_distributor_map_reduce_and_close_are_called(self):
        df = self.create_test_data_sample()

        mock = Mock(spec=DistributorBaseClass)
        mock.close.return_value = None
        mock.map_reduce.return_value = []

        X = extract_features(timeseries_container=df, column_id="id", column_sort="sort", column_kind="kind",
                             column_value="val", default_fc_parameters=self.name_to_param,
                             distributor=mock)

        self.assertTrue(mock.close.called)
        self.assertTrue(mock.map_reduce.called)