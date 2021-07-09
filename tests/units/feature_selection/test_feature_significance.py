# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from unittest import TestCase

import numpy as np
import pandas as pd

from tsfresh.feature_selection.relevance import calculate_relevance_table


class FeatureSignificanceTestCase(TestCase):
    """Test cases for the whole feature selection algorithm."""

    def setUp(self):
        """Fix the random seed."""
        np.random.seed(seed=42)

    def test_binary_target_mixed_case(self):
        # Mixed case with binomial target
        np.random.seed(42)
        y = pd.Series(np.random.binomial(1, 0.5, 1000))
        X = pd.DataFrame(index=range(1000))

        z = y - np.random.binomial(1, 0.1, 1000) + np.random.binomial(1, 0.1, 1000)
        z[z == -1] = 0
        z[z == 2] = 1

        X["rel1"] = z
        X["rel2"] = y * np.abs(np.random.normal(0, 1, 1000)) + np.random.normal(
            0, 1, 1000
        )
        X["rel3"] = y + np.random.normal(0, 0.3, 1000)
        X["rel4"] = y ** 2 + np.random.normal(0, 1, 1000)
        X["rel5"] = np.sqrt(y) + np.random.binomial(2, 0.1, 1000)

        X["irr_constant"] = 1.113344

        X["irr1"] = np.random.normal(0, 1, 1000)
        X["irr2"] = np.random.poisson(1, 1000)
        X["irr3"] = np.random.binomial(1, 0.3, 1000)
        X["irr4"] = np.random.normal(0, 1, 1000)
        X["irr5"] = np.random.poisson(1, 1000)
        X["irr6"] = np.random.binomial(1, 0.3, 1000)
        X["irr7"] = np.random.normal(0, 1, 1000)
        X["irr8"] = np.random.poisson(1, 1000)
        X["irr9"] = np.random.binomial(1, 0.3, 1000)

        df_bh = calculate_relevance_table(X, y)
        feat_rej = df_bh.loc[df_bh.relevant].feature

        # Make sure all selected variables are relevant
        for kept_feature in feat_rej:
            self.assertIn(kept_feature, ["rel1", "rel2", "rel3", "rel4", "rel5"])

        self.assertGreater(len(feat_rej), 0)

        # Test type outputs
        for i in range(1, 6):
            row = df_bh.loc["rel{}".format(i)]
            self.assertEqual(row.feature, "rel{}".format(i))
            if i == 1:
                self.assertEqual(row.type, "binary")
            else:
                self.assertEqual(row.type, "real")

        for i in range(1, 10):
            row = df_bh.loc["irr{}".format(i)]
            self.assertEqual(row.feature, "irr{}".format(i))
            if i not in [3, 6, 9]:
                self.assertEqual(row.type, "real")
            else:
                self.assertEqual(row.type, "binary")

            self.assertFalse(row.relevant)

            # Assert that all of the relevant features are kept.
            # THIS FAILS!
            # self.assertEqual(len(kept_feature), 5)

    def test_binary_target_binary_features(self):
        # Binomial random variables and binomial target
        y = pd.Series(np.random.binomial(1, 0.5, 5000))
        X = pd.DataFrame(index=range(5000))

        for i in range(10):
            X["irr{}".format(i)] = np.random.binomial(1, 0.1, 5000)

        for i in range(10, 20):
            X["irr{}".format(i)] = np.random.binomial(1, 0.8, 5000)

        z = y - np.random.binomial(1, 0.01, 5000) + np.random.binomial(1, 0.01, 5000)
        z[z == -1] = 0
        z[z == 2] = 1
        X["rel1"] = z

        z = y - np.random.binomial(1, 0.05, 5000) + np.random.binomial(1, 0.05, 5000)
        z[z == -1] = 0
        z[z == 2] = 1
        X["rel2"] = z

        z = y - np.random.binomial(1, 0.10, 5000) + np.random.binomial(1, 0.10, 5000)
        z[z == -1] = 0
        z[z == 2] = 1
        X["rel3"] = z

        z = y - np.random.binomial(1, 0.15, 5000) + np.random.binomial(1, 0.15, 5000)
        z[z == -1] = 0
        z[z == 2] = 1
        X["rel4"] = z

        z = y - np.random.binomial(1, 0.20, 5000) + np.random.binomial(1, 0.20, 5000)
        z[z == -1] = 0
        z[z == 2] = 1
        X["rel5"] = z

        df_bh = calculate_relevance_table(X, y)
        feat_rej = df_bh.loc[df_bh.relevant].feature

        # Make sure all selected variables are relevant
        for kept_feature in feat_rej:
            self.assertIn(kept_feature, ["rel1", "rel2", "rel3", "rel4", "rel5"])

        self.assertGreater(len(feat_rej), 0)

        # Test type outputs
        for i in range(1, 6):
            row = df_bh.loc["rel{}".format(i)]
            self.assertEqual(row.feature, "rel{}".format(i))
            self.assertEqual(row.type, "binary")

        for i in range(1, 20):
            row = df_bh.loc["irr{}".format(i)]
            self.assertEqual(row.feature, "irr{}".format(i))
            self.assertEqual(row.type, "binary")

            self.assertFalse(row.relevant)

    def test_binomial_target_realvalued_features(self):
        # Real valued random variables and binomial target
        y = pd.Series(np.random.binomial(1, 0.5, 5000))
        X = pd.DataFrame(index=range(5000))

        for i in range(10):
            X["irr{}".format(i)] = np.random.normal(1, 0.3, 5000)

        for i in range(10, 20):
            X["irr{}".format(i)] = np.random.normal(1, 0.5, 5000)

        for i in range(20, 30):
            X["irr{}".format(i)] = np.random.normal(1, 0.8, 5000)

        X["rel1"] = y * np.random.normal(0, 1, 5000) + np.random.normal(0, 1, 5000)
        X["rel2"] = y + np.random.normal(0, 1, 5000)
        X["rel3"] = y ** 2 + np.random.normal(0, 1, 5000)
        X["rel4"] = np.sqrt(y) + np.random.binomial(2, 0.1, 5000)

        df_bh = calculate_relevance_table(X, y)
        feat_rej = df_bh.loc[df_bh.relevant].feature

        # Make sure all selected variables are relevant
        for kept_feature in feat_rej:
            self.assertIn(kept_feature, ["rel1", "rel2", "rel3", "rel4"])

        self.assertGreater(len(feat_rej), 0)

        # Test type outputs
        for i in range(1, 5):
            row = df_bh.loc["rel{}".format(i)]
            self.assertEqual(row.feature, "rel{}".format(i))
            self.assertEqual(row.type, "real")

        for i in range(1, 30):
            row = df_bh.loc["irr{}".format(i)]
            self.assertEqual(row.feature, "irr{}".format(i))
            self.assertEqual(row.type, "real")

            self.assertFalse(row.relevant)

    def test_real_target_mixed_case(self):
        # Mixed case with real target
        y = pd.Series(np.random.normal(0, 1, 5000))
        X = pd.DataFrame(index=range(5000))

        z = y.copy()
        z[z <= 0] = 0
        z[z > 0] = 1

        X["rel1"] = z
        X["rel2"] = y
        X["rel3"] = y ** 2
        X["rel4"] = np.sqrt(abs(y))

        X["irr1"] = np.random.normal(0, 1, 5000)
        X["irr2"] = np.random.poisson(1, 5000)
        X["irr3"] = np.random.binomial(1, 0.1, 5000)
        X["irr4"] = np.random.normal(0, 1, 5000)
        X["irr5"] = np.random.poisson(1, 5000)
        X["irr6"] = np.random.binomial(1, 0.05, 5000)
        X["irr7"] = np.random.normal(0, 1, 5000)
        X["irr8"] = np.random.poisson(1, 5000)
        X["irr9"] = np.random.binomial(1, 0.2, 5000)

        df_bh = calculate_relevance_table(X, y)
        feat_rej = df_bh.loc[df_bh.relevant].feature

        # Make sure all selected variables are relevant
        for kept_feature in feat_rej:
            self.assertIn(kept_feature, ["rel1", "rel2", "rel3", "rel4"])

        self.assertGreater(len(feat_rej), 0)

        # Test type outputs
        for i in range(1, 5):
            row = df_bh.loc["rel{}".format(i)]
            self.assertEqual(row.feature, "rel{}".format(i))
            if i == 1:
                self.assertEqual(row.type, "binary")
            else:
                self.assertEqual(row.type, "real")

        for i in range(1, 10):
            row = df_bh.loc["irr{}".format(i)]
            self.assertEqual(row.feature, "irr{}".format(i))
            if i in [3, 6, 9]:
                self.assertEqual(row.type, "binary")
            else:
                self.assertEqual(row.type, "real")

            self.assertFalse(row.relevant)

    def test_real_target_binary_features(self):
        # Mixed case with real target
        y = pd.Series(np.random.normal(0, 1, 1000))
        X = pd.DataFrame(index=range(1000))

        z = y - np.random.binomial(1, 0.20, 1000) + np.random.binomial(1, 0.20, 1000)
        z[z == -1] = 0
        z[z == 2] = 1
        X["rel1"] = z

        z = y - np.random.binomial(1, 0.10, 1000) + np.random.binomial(1, 0.10, 1000)
        z[z == -1] = 0
        z[z == 2] = 1
        X["rel2"] = z

        X["irr1"] = np.random.binomial(0, 0.1, 1000)
        X["irr2"] = np.random.binomial(0, 0.15, 1000)
        X["irr3"] = np.random.binomial(0, 0.05, 1000)
        X["irr4"] = np.random.binomial(0, 0.2, 1000)
        X["irr5"] = np.random.binomial(0, 0.25, 1000)
        X["irr6"] = np.random.binomial(0, 0.01, 1000)

        df_bh = calculate_relevance_table(X, y)
        feat_rej = df_bh.loc[df_bh.relevant].feature

        # Make sure all selected variables are relevant
        for kept_feature in feat_rej:
            self.assertIn(kept_feature, ["rel1", "rel2"])

        self.assertGreater(len(feat_rej), 0)

    def test_all_features_good(self):
        # Mixed case with real target
        y = pd.Series(np.random.normal(0, 1, 1000))
        X = pd.DataFrame(index=range(1000))

        z = y - np.random.binomial(1, 0.20, 1000) + np.random.binomial(1, 0.20, 1000)
        z[z == -1] = 0
        z[z == 2] = 1
        X["rel1"] = z

        z = y - np.random.binomial(1, 0.10, 1000) + np.random.binomial(1, 0.10, 1000)
        z[z == -1] = 0
        z[z == 2] = 1
        X["rel2"] = z

        df_bh = calculate_relevance_table(X, y)
        feat_rej = df_bh.loc[df_bh.relevant].feature

        # Make sure all selected variables are relevant
        for kept_feature in feat_rej:
            self.assertIn(kept_feature, ["rel1", "rel2"])

        self.assertGreater(len(feat_rej), 0)

    def test_all_features_bad(self):
        # Mixed case with real target
        y = pd.Series(np.random.normal(0, 1, 1000))
        X = pd.DataFrame(index=range(1000))

        X["irr1"] = np.random.binomial(0, 0.1, 1000)
        X["irr2"] = np.random.binomial(0, 0.15, 1000)
        X["irr3"] = np.random.binomial(0, 0.05, 1000)
        X["irr4"] = np.random.binomial(0, 0.2, 1000)
        X["irr5"] = np.random.binomial(0, 0.25, 1000)
        X["irr6"] = np.random.binomial(0, 0.01, 1000)

        df_bh = calculate_relevance_table(X, y)
        feat_rej = df_bh.loc[df_bh.relevant].feature

        self.assertEqual(len(feat_rej), 0)
