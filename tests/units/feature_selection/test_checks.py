# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import warning_free
from tsfresh.defaults import TEST_FOR_BINARY_TARGET_REAL_FEATURE
from tsfresh.feature_selection.significance_tests import (
    target_binary_feature_binary_test,
    target_binary_feature_real_test,
    target_real_feature_binary_test,
    target_real_feature_real_test,
)


@pytest.fixture()
def binary_series_with_nan():
    return pd.Series([np.NaN, 1, 1])


@pytest.fixture()
def real_series_with_nan():
    return pd.Series([np.NaN, 1, 2])


@pytest.fixture()
def binary_series():
    return pd.Series([0, 1, 1])


@pytest.fixture()
def real_series():
    return pd.Series([0, 1, 2])


class TestChecksBinaryReal:
    """
    Test the checks for the `target_binary_feature_real_test`.
    """

    def test_check_target_is_binary(self, real_series):
        with pytest.raises(ValueError):
            target_binary_feature_real_test(
                x=real_series, y=real_series, test=TEST_FOR_BINARY_TARGET_REAL_FEATURE
            )

    def test_checks_test_function(self, binary_series, real_series):
        with pytest.raises(ValueError):
            target_binary_feature_real_test(
                x=real_series, y=binary_series, test="other_unknown_function"
            )

    def test_checks_feature_nan(self, real_series_with_nan, binary_series):
        with pytest.raises(ValueError):
            target_binary_feature_real_test(
                x=real_series_with_nan,
                y=binary_series,
                test=TEST_FOR_BINARY_TARGET_REAL_FEATURE,
            )

    def test_checks_target_nan(self, binary_series_with_nan, real_series):
        with pytest.raises(ValueError):
            target_binary_feature_real_test(
                x=real_series,
                y=binary_series_with_nan,
                test=TEST_FOR_BINARY_TARGET_REAL_FEATURE,
            )

    def test_check_feature_is_series(self, binary_series, real_series):
        with pytest.raises(TypeError):
            target_binary_feature_real_test(x=real_series.values, y=binary_series)

    def test_check_feature_is_series(self, binary_series, real_series):
        with pytest.raises(TypeError):
            target_binary_feature_real_test(x=real_series, y=binary_series.values)


class TestChecksBinaryBinary:
    """
    Test the checks for the `target_binary_feature_binary_test`.
    """

    def test_checks_feature_is_binary(self, binary_series, real_series):
        with pytest.raises(ValueError):
            target_binary_feature_binary_test(x=real_series, y=binary_series)

    def test_checks_target_is_binary(self, binary_series, real_series):
        with pytest.raises(ValueError):
            target_binary_feature_binary_test(x=binary_series, y=real_series)

    def test_checks_feature_is_series(self, binary_series):
        with pytest.raises(TypeError):
            target_binary_feature_binary_test(x=binary_series.values, y=binary_series)

    def test_checks_target_is_series(self, binary_series):
        with pytest.raises(TypeError):
            target_binary_feature_binary_test(x=binary_series, y=binary_series.values)

    def test_checks_feature_nan(self, binary_series_with_nan, binary_series):
        with pytest.raises(ValueError):
            target_binary_feature_binary_test(x=binary_series_with_nan, y=binary_series)

    def test_checks_target_nan(self, binary_series_with_nan, binary_series):
        with pytest.raises(ValueError):
            target_binary_feature_binary_test(x=binary_series, y=binary_series_with_nan)


class TestChecksRealReal:
    """
    Test the checks for the `target_real_feature_real_test`.
    """

    def test_checks_feature_is_series(self, real_series):
        with pytest.raises(TypeError):
            target_real_feature_real_test(x=real_series.values, y=real_series)

    def test_checks_target_is_series(self, real_series):
        with pytest.raises(TypeError):
            target_real_feature_real_test(x=real_series, y=real_series.values)

    def test_checks_feature_nan(self, real_series_with_nan, real_series):
        with pytest.raises(ValueError):
            target_real_feature_real_test(x=real_series_with_nan, y=real_series)

    def test_checks_target_nan(self, real_series_with_nan, real_series):
        with pytest.raises(ValueError):
            target_real_feature_real_test(x=real_series, y=real_series_with_nan)


class TestChecksRealBinary:
    """
    Test the checks for the `target_real_feature_binary_test`.
    """

    def test_feature_is_binary(self, real_series):
        with pytest.raises(ValueError):
            target_real_feature_binary_test(x=real_series, y=real_series)

    def test_feature_is_series(self, real_series, binary_series):
        with pytest.raises(TypeError):
            target_real_feature_binary_test(x=binary_series.values, y=real_series)

    def test_feature_is_series(self, real_series, binary_series):
        with pytest.raises(TypeError):
            target_real_feature_binary_test(x=binary_series, y=real_series.values)

    def test_checks_feature_nan(self, binary_series_with_nan, real_series):
        with pytest.raises(ValueError):
            target_real_feature_binary_test(x=binary_series_with_nan, y=real_series)

    def test_checks_target_nan(self, real_series_with_nan, binary_series):
        with pytest.raises(ValueError):
            target_real_feature_binary_test(x=binary_series, y=real_series_with_nan)
