# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import pandas as pd
from tests.fixtures import DataTestCase
from tsfresh.feature_extraction import FeatureExtractionSettings
from tsfresh.transformers.relevant_feature_augmenter import RelevantFeatureAugmenter


class RelevantFeatureAugmenterTestCase(DataTestCase):
    def setUp(self):
        self.test_df = self.create_test_data_sample()
        self.extraction_settings = FeatureExtractionSettings()
        self.extraction_settings.set_default_parameters("a")
        calculation_settings_mapping = {
            "length": self.extraction_settings.kind_to_calculation_settings_mapping["a"]["length"]}
        self.extraction_settings.kind_to_calculation_settings_mapping = {"a": calculation_settings_mapping.copy(),
                                                                         "b": calculation_settings_mapping.copy()}

    def test_not_fitted(self):
        augmenter = RelevantFeatureAugmenter()

        X = pd.DataFrame()

        self.assertRaises(RuntimeError, augmenter.transform, X)

    def test_no_timeseries(self):
        augmenter = RelevantFeatureAugmenter()

        X = pd.DataFrame()
        y = pd.Series()

        self.assertRaises(RuntimeError, augmenter.fit, X, y)

    def test_nothing_relevant(self):
        augmenter = RelevantFeatureAugmenter(feature_extraction_settings=self.extraction_settings,
                                            column_value="val", column_id="id", column_sort="sort",
                                            column_kind="kind")

        y = pd.Series({10: 1, 500: 0})
        X = pd.DataFrame(index=[10, 500])

        augmenter.set_timeseries_container(self.test_df)
        augmenter.fit(X, y)

        transformed_X = augmenter.transform(X.copy())

        self.assertEqual(list(transformed_X.columns), [])
        self.assertEqual(list(transformed_X.index), list(X.index))

    def test_impute_works(self):
        self.extraction_settings.kind_to_calculation_settings_mapping["a"].update({"kurtosis": None})

        augmeter = RelevantFeatureAugmenter(feature_extraction_settings=self.extraction_settings,
                                            column_value="val", column_id="id", column_sort="sort",
                                            column_kind="kind")

        y = pd.Series({10: 1, 500: 0})
        X = pd.DataFrame(index=[10, 500])

        augmeter.set_timeseries_container(self.test_df)
        augmeter.fit(X, y)

        transformed_X = augmeter.transform(X.copy())

        self.assertEqual(list(transformed_X.columns), [])
        self.assertEqual(list(transformed_X.index), list(X.index))


    # def test_with_numpy_array(self):
    #     selector = FeatureSelector()
    #
    #     y = pd.Series(np.random.binomial(1, 0.5, 1000))
    #     X = pd.DataFrame(index=range(1000))
    #
    #     X["irr1"] = np.random.normal(0, 1, 1000)
    #     X["rel1"] = y
    #
    #     y_numpy = y.values
    #     X_numpy = X.as_matrix()
    #
    #     selector.fit(X, y)
    #     selected_X = selector.transform(X)
    #
    #     selector.fit(X_numpy, y_numpy)
    #     selected_X_numpy = selector.transform(X_numpy)
    #
    #     self.assertTrue((selected_X_numpy == selected_X.values).all())
    #
    #     self.assertTrue(selected_X_numpy.shape, (1, 1000))
