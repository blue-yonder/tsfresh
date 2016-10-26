# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from __future__ import absolute_import, division
import numpy as np
import pandas as pd
from tests.fixtures import DataTestCase
from tsfresh.feature_extraction.extraction import extract_features
from tsfresh.feature_extraction.settings import FeatureExtractionSettings


class FeatureExtractorTestCase(DataTestCase):
    """The unit tests in this module make sure if the time series features are created properly"""

    def setUp(self):
        self.settings = FeatureExtractionSettings()
        self.settings.PROFILING = False

    def test_calculate_ts_features(self):
        # todo: implement more methods and test more aspects
        df = self.create_test_data_sample()
        extracted_features = extract_features(df, self.settings, "id", "sort", "kind", "val")

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
        extracted_features_sts = extract_features(df_sts, self.settings, "id", "sort", "kind", "val")

        self.assertIsInstance(extracted_features_sts, pd.DataFrame)
        self.assertTrue(np.all(extracted_features_sts.a__maximum == np.array([1.0, 6.0])))
        self.assertTrue(np.all(extracted_features_sts.a__sum_values == np.array([1.0, 11.0])))
        self.assertTrue(np.all(extracted_features_sts.a__count_above_mean == np.array([0, 1])))

    def test_calculate_ts_features_after_randomisation(self):
        df = self.create_test_data_sample()
        df_random = df.copy().sample(frac=1)

        extracted_features = extract_features(df, self.settings, "id", "sort", "kind", "val").sort_index()
        extracted_features_from_random = extract_features(df_random, self.settings,
                                                          "id", "sort", "kind", "val").sort_index()

        self.assertItemsEqual(extracted_features.columns, extracted_features_from_random.columns)

        for col in extracted_features:
            self.assertIsNone(np.testing.assert_array_almost_equal(extracted_features[col],
                                                                   extracted_features_from_random[col]))
