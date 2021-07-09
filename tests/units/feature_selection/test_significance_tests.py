# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
import numpy as np
import pandas as pd
import pytest

from tsfresh.defaults import TEST_FOR_BINARY_TARGET_REAL_FEATURE
from tsfresh.feature_selection.significance_tests import (
    target_binary_feature_binary_test,
    target_binary_feature_real_test,
    target_real_feature_binary_test,
    target_real_feature_real_test,
)


@pytest.fixture()
def set_random_seed():
    np.random.seed(seed=42)


@pytest.fixture()
def binary_feature(set_random_seed):
    return pd.Series(np.random.binomial(1, 0.5, 250), name="TEST")


@pytest.fixture()
def binary_target_not_related(set_random_seed):
    return pd.Series(np.random.binomial(1, 0.5, 250))


@pytest.fixture()
def real_feature(set_random_seed):
    return pd.Series(np.random.normal(0, 1, 250), name="TEST")


@pytest.fixture()
def real_target_not_related(set_random_seed):
    return pd.Series(np.random.normal(0, 1, 250))


class TestUnsignificant:
    @pytest.fixture()
    def minimal_p_value_for_unsignificant_features(self):
        return 0.05

    def test_feature_selection_target_binary_features_binary(
        self,
        minimal_p_value_for_unsignificant_features,
        binary_feature,
        binary_target_not_related,
    ):
        """
        Test if the p_value returned by target_binary_feature_binary_test is
        large enough for highly unsignificant features.
        """
        p_value = target_binary_feature_binary_test(
            binary_feature, binary_target_not_related
        )
        assert minimal_p_value_for_unsignificant_features < p_value

    def test_feature_selection_target_binary_features_realvalued(
        self,
        minimal_p_value_for_unsignificant_features,
        real_feature,
        binary_target_not_related,
    ):
        """
        Test if the p_value returned by target_binary_feature_binary_test is
        large enough for highly unsignificant features.
        """
        p_value = target_binary_feature_real_test(
            real_feature, binary_target_not_related, TEST_FOR_BINARY_TARGET_REAL_FEATURE
        )
        assert minimal_p_value_for_unsignificant_features < p_value

    def test_feature_selection_target_realvalued_features_binary(
        self,
        minimal_p_value_for_unsignificant_features,
        binary_feature,
        real_target_not_related,
    ):
        """
        Test if the p_value returned by target_real_feature_binary_test is
        large enough for highly unsignificant features."""
        p_value = target_real_feature_binary_test(
            binary_feature, real_target_not_related
        )
        assert minimal_p_value_for_unsignificant_features < p_value

    def test_feature_selection_target_realvalued_features_realvalued(
        self,
        minimal_p_value_for_unsignificant_features,
        real_feature,
        real_target_not_related,
    ):
        """
        Test if the p_value returned by target_real_feature_real_test is
        large enough for highly unsignificant features.
        """
        p_value = target_real_feature_real_test(real_feature, real_target_not_related)
        assert minimal_p_value_for_unsignificant_features < p_value


class TestSignificant:
    @pytest.fixture()
    def maximal_p_value_for_significant_features(self):
        return 0.15

    def test_feature_selection_target_binary_features_binary(
        self, maximal_p_value_for_significant_features, binary_feature
    ):
        """
        Test if the p_value returned by target_binary_feature_binary_test is
        low enough for highly significant features.
        """
        y = binary_feature - pd.Series(
            np.random.binomial(1, 0.1, 250) + np.random.binomial(1, 0.1, 250)
        )
        y[y == -1] = 0
        y[y == -2] = 0
        y[y == 2] = 1

        p_value = target_binary_feature_binary_test(binary_feature, y)
        assert maximal_p_value_for_significant_features > p_value

    def test_feature_selection_target_binary_features_realvalued_mann(
        self, maximal_p_value_for_significant_features, real_feature
    ):
        """
        Test if the p_value returned by target_binary_feature_real_test is
        low enough for highly significant features.
        """
        y = pd.Series(np.ndarray(250))
        y[real_feature >= 0.3] = 1
        y[real_feature < 0.3] = 0
        y -= pd.Series(np.random.binomial(1, 0.1, 250))
        y[y == -1] = 0
        y[y == 2] = 1

        p_value = target_binary_feature_real_test(
            real_feature, y, TEST_FOR_BINARY_TARGET_REAL_FEATURE
        )
        assert maximal_p_value_for_significant_features > p_value

    def test_feature_selection_target_binary_features_realvalued_smir(
        self, maximal_p_value_for_significant_features, real_feature
    ):
        """
        Test if the p_value returned by target_binary_feature_real_test is
        low enough for highly significant features.
        """
        y = pd.Series(np.ndarray(250))
        y[real_feature >= 0.3] = 1
        y[real_feature < 0.3] = 0
        y -= pd.Series(np.random.binomial(1, 0.2, 250))
        y[y == -1] = 0
        y[y == 2] = 1

        p_value = target_binary_feature_real_test(real_feature, y, test="smir")
        assert maximal_p_value_for_significant_features > p_value

    def test_feature_selection_target_realvalued_features_binary(
        self, maximal_p_value_for_significant_features, binary_feature
    ):
        """
        Test if the p_value returned by target_real_feature_binary_test is
        low enough for highly significant features.
        """
        y = binary_feature * pd.Series(np.random.normal(0, 1, 250)) + pd.Series(
            np.random.normal(0, 0.25, 250)
        )

        p_value = target_real_feature_binary_test(binary_feature, y)
        assert maximal_p_value_for_significant_features > p_value

    def test_feature_selection_target_realvalued_features_realvalued(
        self, maximal_p_value_for_significant_features, real_feature
    ):
        """
        Test if the p_value returned by target_real_feature_real_test is
        low enough for highly significant features.
        """
        y = real_feature + pd.Series(np.random.normal(0, 1, 250))

        p_value = target_real_feature_real_test(real_feature, y)

        assert maximal_p_value_for_significant_features > p_value
