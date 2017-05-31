# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import pandas as pd
import numpy as np
from tests.fixtures import DataTestCase
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.transformers.relevant_feature_augmenter import RelevantFeatureAugmenter


class RelevantFeatureAugmenterTestCase(DataTestCase):
    def setUp(self):
        self.test_df = self.create_test_data_sample()
        fc_parameters = {"length": None}
        self.kind_to_fc_parameters = {"a": fc_parameters.copy(),
                                      "b": fc_parameters.copy()}

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
        augmenter = RelevantFeatureAugmenter(kind_to_fc_parameters=self.kind_to_fc_parameters,
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
        self.kind_to_fc_parameters["a"].update({"kurtosis": None})

        augmeter = RelevantFeatureAugmenter(kind_to_fc_parameters=self.kind_to_fc_parameters,
                                            column_value="val", column_id="id", column_sort="sort",
                                            column_kind="kind")

        y = pd.Series({10: 1, 500: 0})
        X = pd.DataFrame(index=[10, 500])

        augmeter.set_timeseries_container(self.test_df)
        augmeter.fit(X, y)

        transformed_X = augmeter.transform(X.copy())

        self.assertEqual(list(transformed_X.columns), [])
        self.assertEqual(list(transformed_X.index), list(X.index))

    def test_evaluate_only_added_features_true(self):
        """
        The boolean flag `evaluate_only_extracted_features` makes sure that only the time series based features are
        filtered. This unit tests checks that
        """

        augmenter = RelevantFeatureAugmenter(kind_to_fc_parameters=self.kind_to_fc_parameters,
                                             filter_only_tsfresh_features=True,
                                             column_value="val", column_id="id", column_sort="sort", column_kind="kind")

        y = pd.Series({10: 1, 500: 0})
        X = pd.DataFrame(index=[10, 500])
        X["pre_feature"] = 0

        augmenter.set_timeseries_container(self.test_df)
        augmenter.fit(X, y)
        transformed_X = augmenter.transform(X.copy())

        self.assertEqual(sum(["pre_feature" == column for column in transformed_X.columns]), 1)

    def test_evaluate_only_added_features_false(self):
        """
        The boolean flag `evaluate_only_extracted_features` makes sure that only the time series based features are
        filtered. This unit tests checks that
        """

        augmenter = RelevantFeatureAugmenter(kind_to_fc_parameters=self.kind_to_fc_parameters,
                                             filter_only_tsfresh_features=False,
                                             column_value="val", column_id="id", column_sort="sort", column_kind="kind")

        df, y = self.create_test_data_sample_with_target()
        X = pd.DataFrame(index=np.unique(df.id))
        X["pre_drop"] = 0
        X["pre_keep"] = y

        augmenter.set_timeseries_container(df)
        augmenter.fit(X, y)
        transformed_X = augmenter.transform(X.copy())

        self.assertEqual(sum(["pre_keep" == column for column in transformed_X.columns]), 1)
        self.assertEqual(sum(["pre_drop" == column for column in transformed_X.columns]), 0)
