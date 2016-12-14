# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import numpy as np
from unittest import TestCase
import pandas as pd
import tsfresh.feature_selection.significance_tests
import tsfresh.feature_selection.feature_selector

# the unit tests in this module make sure if obvious irrelevant features are rejected by the feature
# selection algorithms
from tsfresh.feature_selection.settings import FeatureSignificanceTestsSettings


# noinspection PyUnresolvedReferences
class FeatureSelection(TestCase):
    """
    Test cases for selection of significant and rejection of unsignificant features.
    """

    def setUp(self):
        """Set the random seed to 42."""
        TestCase.setUp(self)
        np.random.seed(seed=42)

        self.minimal_p_value_for_unsignificant_features = 0.05
        self.maximal_p_value_for_significant_features = 0.15

        self.settings = FeatureSignificanceTestsSettings()

    def test_feature_selection_target_binary_features_binary_greater(self):
        """Test if the p_value returned by target_binary_feature_binary_test is
        large enough for highly unsignificant features."""
        x = pd.Series(np.random.binomial(1, 0.5, 250), name="TEST")
        y = pd.Series(np.random.binomial(1, 0.5, 250))

        p_value = tsfresh.feature_selection.significance_tests.target_binary_feature_binary_test(x, y)
        self.assertGreater(p_value, self.minimal_p_value_for_unsignificant_features)

        result_series = tsfresh.feature_selection.feature_selector._calculate_p_value(x, y, self.settings, True)
        self.assertEqual(result_series.name, "TEST")
        self.assertGreater(result_series.p_value, self.minimal_p_value_for_unsignificant_features)
        self.assertEqual(result_series.type, "binary")

    def test_feature_selection_target_binary_features_realvalued_greater(self):
        """Test if the p_value returned by target_binary_feature_binary_test is
        large enough for highly unsignificant features."""
        x = pd.Series(np.random.normal(0, 1, 250), name="TEST")
        y = pd.Series(np.random.binomial(1, 0.5, 250))

        p_value = tsfresh.feature_selection.significance_tests.target_binary_feature_real_test(x, y, self.settings)

        self.assertGreater(p_value, self.minimal_p_value_for_unsignificant_features)

        result_series = tsfresh.feature_selection.feature_selector._calculate_p_value(x, y, self.settings, True)
        self.assertEqual(result_series.name, "TEST")
        self.assertGreater(result_series.p_value, self.minimal_p_value_for_unsignificant_features)
        self.assertEqual(result_series.type, "real")

    def test_feature_selection_target_realvalued_features_binary_greater(self):
        """Test if the p_value returned by target_real_feature_binary_test is
        large enough for highly unsignificant features."""
        x = pd.Series(np.random.binomial(1, 0.5, 250), name="TEST")
        y = pd.Series(np.random.normal(0, 1, 250))

        p_value = tsfresh.feature_selection.significance_tests.target_real_feature_binary_test(x, y)

        self.assertGreater(p_value, self.minimal_p_value_for_unsignificant_features)

        result_series = tsfresh.feature_selection.feature_selector._calculate_p_value(x, y, self.settings, False)
        self.assertEqual(result_series.name, "TEST")
        self.assertGreater(result_series.p_value, self.minimal_p_value_for_unsignificant_features)
        self.assertEqual(result_series.type, "binary")

    def test_feature_selection_target_realvalued_features_realvalued_greater(self):
        """Test if the p_value returned by target_real_feature_real_test is
        large enough for highly unsignificant features."""
        x = pd.Series(np.random.normal(0, 1, 250), name="TEST")
        y = pd.Series(np.random.normal(0, 1, 250))

        p_value = tsfresh.feature_selection.significance_tests.target_real_feature_real_test(x, y)

        self.assertGreater(p_value, self.minimal_p_value_for_unsignificant_features)

        result_series = tsfresh.feature_selection.feature_selector._calculate_p_value(x, y, self.settings, False)
        self.assertEqual(result_series.name, "TEST")
        self.assertGreater(result_series.p_value, self.minimal_p_value_for_unsignificant_features)
        self.assertEqual(result_series.type, "real")

    def test_feature_selection_target_binary_features_binary_less(self):
        """Test if the p_value returned by target_binary_feature_binary_test is
        low enough for highly significant features."""
        x = pd.Series(np.random.binomial(1, 0.5, 250), name="TEST")
        y = x - pd.Series(np.random.binomial(1, 0.1, 250) + np.random.binomial(1, 0.1, 250))
        y[y == -1] = 0
        y[y == -2] = 0
        y[y == 2] = 1

        p_value = tsfresh.feature_selection.significance_tests.target_binary_feature_binary_test(x, y)
        self.assertLess(p_value, self.maximal_p_value_for_significant_features)

        result_series = tsfresh.feature_selection.feature_selector._calculate_p_value(x, y, self.settings, True)
        self.assertEqual(result_series.name, "TEST")
        self.assertLess(result_series.p_value, self.maximal_p_value_for_significant_features)
        self.assertEqual(result_series.type, "binary")

    def test_feature_selection_target_binary_features_realvalued_mann_less(self):
        """Test if the p_value returned by target_binary_feature_real_test is
        low enough for highly significant features."""
        x = pd.Series(np.random.normal(0, 1, 250), name="TEST")
        y = pd.Series(np.ndarray(250))
        y[x >= 0.3] = 1
        y[x < 0.3] = 0
        y -= pd.Series(np.random.binomial(1, 0.1, 250))
        y[y == -1] = 0
        y[y == 2] = 1

        p_value = tsfresh.feature_selection.significance_tests.target_binary_feature_real_test(x, y, self.settings)
        self.assertLess(p_value, self.maximal_p_value_for_significant_features)

        result_series = tsfresh.feature_selection.feature_selector._calculate_p_value(x, y, self.settings, True)
        self.assertEqual(result_series.name, "TEST")
        self.assertLess(result_series.p_value, self.maximal_p_value_for_significant_features)
        self.assertEqual(result_series.type, "real")

    def test_feature_selection_target_binary_features_realvalued_smir_less(self):
        """Test if the p_value returned by target_binary_feature_real_test is
        low enough for highly significant features."""
        tmp_settings = self.settings
        tmp_settings.test_for_binary_target_real_feature = "smir"

        x = pd.Series(np.random.normal(0, 1, 250), name="TEST")
        y = pd.Series(np.ndarray(250))
        y[x >= 0.3] = 1
        y[x < 0.3] = 0
        y -= pd.Series(np.random.binomial(1, 0.2, 250))
        y[y == -1] = 0
        y[y == 2] = 1

        p_value = tsfresh.feature_selection.significance_tests.target_binary_feature_real_test(x, y, tmp_settings)
        self.assertLess(p_value, self.maximal_p_value_for_significant_features)

        result_series = tsfresh.feature_selection.feature_selector._calculate_p_value(x, y, self.settings, True)
        self.assertEqual(result_series.name, "TEST")
        self.assertLess(result_series.p_value, self.maximal_p_value_for_significant_features)
        self.assertEqual(result_series.type, "real")

    def test_feature_selection_target_realvalued_features_binary_less(self):
        """Test if the p_value returned by target_real_feature_binary_test is
        low enough for highly significant features."""
        x = pd.Series(np.random.binomial(1, 0.5, 250), name="TEST")
        y = x * pd.Series(np.random.normal(0, 1, 250)) + pd.Series(np.random.normal(0, 0.25, 250))

        p_value = tsfresh.feature_selection.significance_tests.target_real_feature_binary_test(x, y)
        self.assertLess(p_value, self.maximal_p_value_for_significant_features)

        result_series = tsfresh.feature_selection.feature_selector._calculate_p_value(x, y, self.settings, False)
        self.assertEqual(result_series.name, "TEST")
        self.assertLess(result_series.p_value, self.maximal_p_value_for_significant_features)
        self.assertEqual(result_series.type, "binary")

    def test_feature_selection_target_realvalued_features_realvalued_less(self):
        """Test if the p_value returned by target_real_feature_real_test is
        low enough for highly significant features."""
        x = pd.Series(np.random.normal(0, 1, 250), name="TEST")
        y = x + pd.Series(np.random.normal(0, 1, 250))

        p_value = tsfresh.feature_selection.significance_tests.target_real_feature_real_test(x, y)

        self.assertLess(p_value, self.maximal_p_value_for_significant_features)

        result_series = tsfresh.feature_selection.feature_selector._calculate_p_value(x, y, self.settings, False)
        self.assertEqual(result_series.name, "TEST")
        self.assertLess(result_series.p_value, self.maximal_p_value_for_significant_features)
        self.assertEqual(result_series.type, "real")

    def test_const_feature(self):
        """Constant features should not be used in the p-value calculation."""
        x = pd.Series([1.111]*250, name="TEST")
        y = x + pd.Series(np.random.normal(0, 1, 250))

        result_series = tsfresh.feature_selection.feature_selector._calculate_p_value(x, y, self.settings, False)
        self.assertEqual(result_series.name, "TEST")
        self.assertEqual(result_series.type, "const")
        self.assertEqual(result_series.rejected, False)


# noinspection PyUnresolvedReferences
class FeatureSelectionConfigTestCase(TestCase):
    """
    Test cases for the configuration and type tests of the feature selectors.
    """

    def setUp(self):
        self.settings = FeatureSignificanceTestsSettings()

    def test_fs_tb_fb_binary(self):
        self.assertRaises(ValueError, tsfresh.feature_selection.significance_tests.target_binary_feature_binary_test,
                          x=pd.Series([0, 1, 2]),
                          y=pd.Series([0, 1, 1]))
        self.assertRaises(ValueError, tsfresh.feature_selection.significance_tests.target_binary_feature_binary_test,
                          x=pd.Series([0, 1, 1]),
                          y=pd.Series([0, 1, 2]))

        # Should not fail
        tsfresh.feature_selection.significance_tests.target_binary_feature_binary_test(x=pd.Series([1, 2, 1]),
                                                                                       y=pd.Series([0, 2, 0]))

    def test_fs_tb_fr_binary(self):
        self.assertRaises(ValueError, tsfresh.feature_selection.significance_tests.target_binary_feature_real_test,
                          x=pd.Series([0, 1, 2]),
                          y=pd.Series([0, 1, 2]), settings=self.settings)

        # Should not fail
        tsfresh.feature_selection.significance_tests.target_binary_feature_real_test(x=pd.Series([0, 1, 2]),
                                                                                     y=pd.Series([0, 2, 0]),
                                                                                     settings=self.settings)

    def test_fs_tr_fb_binary(self):
        self.assertRaises(ValueError, tsfresh.feature_selection.significance_tests.target_real_feature_binary_test,
                          x=pd.Series([0, 1, 2]),
                          y=pd.Series([0, 1, 2]))

        tsfresh.feature_selection.significance_tests.target_real_feature_binary_test(x=pd.Series([0, 2, 0]),
                                                                                     y=pd.Series([0, 1, 2]))

    def test_fs_tb_fr_config(self):
        self.settings.test_for_binary_target_real_feature = "other_unknown_function"

        # Unneeded data (the function call will fail probably)
        x = pd.Series(np.random.normal(0, 1, 250), name="TEST")
        y = pd.Series(np.random.binomial(1, 0.5, 250))

        self.assertRaises(ValueError, tsfresh.feature_selection.significance_tests.target_binary_feature_real_test, x=x,
                          y=y,
                          settings=self.settings)

    def test_fs_tb_fb_series(self):
        self.assertRaises(TypeError, tsfresh.feature_selection.significance_tests.target_binary_feature_binary_test,
                          x=[0, 1, 2],
                          y=pd.Series([0, 1, 2]))
        self.assertRaises(TypeError, tsfresh.feature_selection.significance_tests.target_binary_feature_binary_test,
                          x=pd.Series([0, 1, 2]),
                          y=[0, 1, 2])

    def test_fs_tr_fb_series(self):
        self.assertRaises(TypeError, tsfresh.feature_selection.significance_tests.target_real_feature_binary_test,
                          x=[0, 1, 2],
                          y=pd.Series([0, 1, 2]))
        self.assertRaises(TypeError, tsfresh.feature_selection.significance_tests.target_real_feature_binary_test,
                          x=pd.Series([0, 1, 2]),
                          y=[0, 1, 2])

    def test_fs_tb_fr_series(self):
        self.assertRaises(TypeError, tsfresh.feature_selection.significance_tests.target_binary_feature_real_test,
                          x=[0, 1, 2],
                          y=pd.Series([0, 1, 2]))
        self.assertRaises(TypeError, tsfresh.feature_selection.significance_tests.target_binary_feature_real_test,
                          x=pd.Series([0, 1, 2]),
                          y=[0, 1, 2])

    def test_fs_tr_fr_series(self):
        self.assertRaises(TypeError, tsfresh.feature_selection.significance_tests.target_real_feature_real_test,
                          x=[0, 1, 2],
                          y=pd.Series([0, 1, 2]))
        self.assertRaises(TypeError, tsfresh.feature_selection.significance_tests.target_real_feature_real_test,
                          x=pd.Series([0, 1, 2]),
                          y=[0, 1, 2])
