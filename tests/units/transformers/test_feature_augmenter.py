# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import numpy as np
import pandas as pd

from tests.fixtures import DataTestCase
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.transformers import FeatureAugmenter


class FeatureAugmenterTestCase(DataTestCase):
    def setUp(self):
        self.test_df = self.create_test_data_sample()

        fc_parameters = {"length": None}
        self.kind_to_fc_parameters = {
            "a": fc_parameters.copy(),
            "b": fc_parameters.copy(),
        }

    def test_fit_and_transform(self):
        augmenter = FeatureAugmenter(
            column_value="val",
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            n_jobs=0,
            disable_progressbar=True,
        )

        # Fit should do nothing
        returned_df = augmenter.fit()
        self.assertCountEqual(returned_df.__dict__, augmenter.__dict__)
        self.assertRaises(RuntimeError, augmenter.transform, None)

        augmenter.set_timeseries_container(self.test_df)

        # Add features to all time series
        X_with_index = pd.DataFrame([{"feature_1": 1}] * 2, index=[10, 500])
        X_transformed = augmenter.transform(X_with_index)

        # Require same shape
        for i in X_transformed.index:
            self.assertIn(i, X_with_index.index)

        for i in X_with_index.index:
            self.assertIn(i, X_transformed.index)

        self.assertEqual(X_transformed.shape, (2, 3))

        # Preserve old features
        self.assertCountEqual(
            list(X_transformed.columns), ["feature_1", "a__length", "b__length"]
        )

        # Features are not allowed to be NaN
        for index, row in X_transformed.iterrows():
            print((index, row))
            self.assertFalse(np.isnan(row["a__length"]))
            self.assertFalse(np.isnan(row["b__length"]))

    def test_add_features_to_only_a_part(self):
        augmenter = FeatureAugmenter(
            column_value="val",
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            n_jobs=0,
            disable_progressbar=True,
        )

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

    def test_no_ids_present(self):
        augmenter = FeatureAugmenter(
            column_value="val",
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            n_jobs=0,
            disable_progressbar=True,
        )

        augmenter.set_timeseries_container(self.test_df)

        X_with_not_all_ids = pd.DataFrame([{"feature_1": 1}], index=[-999])
        self.assertRaisesRegex(
            AttributeError,
            r"The ids of the time series container",
            augmenter.transform,
            X_with_not_all_ids,
        )
