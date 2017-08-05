# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import pytest
import pandas as pd
import numpy as np

from tsfresh.defaults import TEST_FOR_BINARY_TARGET_REAL_FEATURE
from tsfresh.feature_selection.significance_tests import target_real_feature_binary_test, target_real_feature_real_test,\
    target_binary_feature_real_test, target_binary_feature_binary_test


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
            target_binary_feature_real_test(x=real_series, y=real_series,
                                            test=TEST_FOR_BINARY_TARGET_REAL_FEATURE)

    def test_checks_test_function(self, binary_series, real_series):
        with pytest.raises(ValueError):
            target_binary_feature_real_test(x=real_series, y=binary_series, test="other_unknown_function")

    def test_checks_feature_nan(self, real_series_with_nan, binary_series):
        with pytest.raises(ValueError):
            target_binary_feature_real_test(x=real_series_with_nan, y=binary_series,
                                            test=TEST_FOR_BINARY_TARGET_REAL_FEATURE)

    def test_checks_target_nan(self, binary_series_with_nan, real_series):
        with pytest.raises(ValueError):
            target_binary_feature_real_test(x=real_series, y=binary_series_with_nan,
                                            test=TEST_FOR_BINARY_TARGET_REAL_FEATURE)


class TestFeatureSignificanceTestsChecks:
    """
    Test cases for the configuration and type tests of the feature selectors.
    """


    def test_fs_tb_fb_binary(self):
        with pytest.raises(ValueError):
            target_binary_feature_binary_test(x=pd.Series([0, 1, 2]), y=pd.Series([0, 1, 1]))

        with pytest.raises(ValueError):
            target_binary_feature_binary_test(x=pd.Series([0, 1, 1]), y=pd.Series([0, 1, 2]))

        # Should not fail
        target_binary_feature_binary_test(x=pd.Series([1, 2, 1]), y=pd.Series([0, 2, 0]))



    def test_fs_tr_fb_binary(self):
        with pytest.raises(ValueError):
            target_real_feature_binary_test(x=pd.Series([0, 1, 2]), y=pd.Series([0, 1, 2]))

        target_real_feature_binary_test(x=pd.Series([0, 2, 0]), y=pd.Series([0, 1, 2]))



    def test_fs_tb_fb_series(self):
        with pytest.raises(TypeError):
            target_binary_feature_binary_test(x=[0, 1, 2], y=pd.Series([0, 1, 2]))

        with pytest.raises(TypeError):
            target_binary_feature_binary_test(x=pd.Series([0, 1, 2]), y=[0, 1, 2])

    def test_fs_tr_fb_series(self):
        with pytest.raises(TypeError):
            target_real_feature_binary_test(x=[0, 1, 2], y=pd.Series([0, 1, 2]))
        with pytest.raises(TypeError):
            target_real_feature_binary_test(x=pd.Series([0, 1, 2]), y=[0, 1, 2])

    def test_fs_tb_fr_series(self):
        with pytest.raises(TypeError):
            target_binary_feature_real_test(x=[0, 1, 2], y=pd.Series([0, 1, 2]))

        with pytest.raises(TypeError):
            target_binary_feature_real_test(x=pd.Series([0, 1, 2]), y=[0, 1, 2])

    def test_fs_tr_fr_series(self):
        with pytest.raises(TypeError):
            target_real_feature_real_test(x=[0, 1, 2], y=pd.Series([0, 1, 2]))

        with pytest.raises(TypeError):
            target_real_feature_real_test(x=pd.Series([0, 1, 2]), y=[0, 1, 2])

    def test_fb_tb_feature_nan(self, binary_series_with_nan, binary_series):
        with pytest.raises(ValueError):
            target_binary_feature_binary_test(x=binary_series_with_nan, y=binary_series)

    def test_fb_tb_target_nan(self, binary_series_with_nan, binary_series):
        with pytest.raises(ValueError):
            target_binary_feature_binary_test(x=binary_series, y=binary_series_with_nan)

    def test_fb_tr_feature_nan(self, binary_series_with_nan, real_series):
        with pytest.raises(ValueError):
            target_real_feature_binary_test(x=binary_series_with_nan, y=real_series)

    def test_fb_tr_target_nan(self, real_series_with_nan, binary_series):
        with pytest.raises(ValueError):
            target_real_feature_binary_test(x=binary_series, y=real_series_with_nan)



    def test_fr_tr_feature_nan(self, real_series_with_nan, real_series):
        with pytest.raises(ValueError):
            target_real_feature_real_test(x=real_series_with_nan, y=real_series)

    def test_fr_tr_target_nan(self, real_series_with_nan, real_series):
        with pytest.raises(ValueError):
            target_real_feature_real_test(x=real_series, y=real_series_with_nan)
