# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from __future__ import print_function
import pandas as pd
from tests.fixtures import DataTestCase
from tsfresh.feature_extraction.settings import FeatureExtractionSettings
from tsfresh.transformers import FeatureAugmenter
import six
import numpy as np

class FeatureAugmenterTestCase(DataTestCase):
    def setUp(self):
        self.test_df = self.create_test_data_sample()
        self.settings = FeatureExtractionSettings()
        self.settings.set_default_parameters("a")
        calculation_settings_mapping = {"length": self.settings.kind_to_calculation_settings_mapping["a"]["length"]}
        self.settings.kind_to_calculation_settings_mapping = {"a": calculation_settings_mapping.copy(),
                                                              "b": calculation_settings_mapping.copy()}

    def test_fit_and_transform(self):
        augmenter = FeatureAugmenter(column_value="val", column_id="id", column_sort="sort",
                                     column_kind="kind", settings=self.settings)

        # Fit should do nothing
        returned_df = augmenter.fit()
        six.assertCountEqual(self, returned_df.__dict__, augmenter.__dict__)
        self.assertRaises(RuntimeError, augmenter.transform, None)

        augmenter.set_timeseries_container(self.test_df)

        # Add features to all time series
        X_with_index = pd.DataFrame([{"feature_1": 1}]*2, index=[10, 500])
        X_transformed = augmenter.transform(X_with_index)

        # Require same shape
        for i in X_transformed.index:
            self.assertIn(i, X_with_index.index)

        for i in X_with_index.index:
            self.assertIn(i, X_transformed.index)

        self.assertEqual(X_transformed.shape, (2, 3))

        # Preserve old features
        six.assertCountEqual(self, list(X_transformed.columns), ["feature_1", "a__length", "b__length"])

        # Features are not allowed to be NaN
        for index, row in X_transformed.iterrows():
            print((index, row))
            self.assertFalse(np.isnan(row["a__length"]))
            self.assertFalse(np.isnan(row["b__length"]))

    def test_add_features_to_only_a_part(self):
        augmenter = FeatureAugmenter(column_value="val", column_id="id", column_sort="sort",
                                     column_kind="kind", settings=self.settings)

        augmenter.set_timeseries_container(self.test_df)

        X_with_not_all_ids = pd.DataFrame([{"feature_1": 1}], index=[10])
        X_transformed = augmenter.transform(X_with_not_all_ids)

        for i in X_transformed.index:
            self.assertIn(i, X_with_not_all_ids.index)

        for i in X_with_not_all_ids.index:
            self.assertIn(i, X_transformed.index)

        self.assertEqual(X_transformed.shape, (1, 3))
        self.assertEqual(X_transformed.index, [10])

        # Features are not allowed to be NaN
        for index, row in X_transformed.iterrows():
            print((index, row))
            self.assertFalse(np.isnan(row["a__length"]))
            self.assertFalse(np.isnan(row["b__length"]))
