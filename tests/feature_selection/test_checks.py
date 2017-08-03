# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import pytest
import pandas as pd
import numpy as np

from tsfresh.defaults import TEST_FOR_BINARY_TARGET_REAL_FEATURE
from tsfresh.feature_selection.significance_tests import target_real_feature_binary_test, target_real_feature_real_test,\
    target_binary_feature_real_test, target_binary_feature_binary_test


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

    def test_fs_tb_fr_binary(self):
        with pytest.raises(ValueError):
            target_binary_feature_real_test(x=pd.Series([0, 1, 2]), y=pd.Series([0, 1, 2]),
                                            test=TEST_FOR_BINARY_TARGET_REAL_FEATURE)

        # Should not fail
        target_binary_feature_real_test(x=pd.Series([0, 1, 2]), y=pd.Series([0, 2, 0]),
                                        test=TEST_FOR_BINARY_TARGET_REAL_FEATURE)

    def test_fs_tr_fb_binary(self):
        with pytest.raises(ValueError):
            target_real_feature_binary_test(x=pd.Series([0, 1, 2]), y=pd.Series([0, 1, 2]))

        target_real_feature_binary_test(x=pd.Series([0, 2, 0]), y=pd.Series([0, 1, 2]))

    def test_fs_tb_fr_config(self):
        # Unneeded data (the function call will fail probably)
        x = pd.Series(np.random.normal(0, 1, 250), name="TEST")
        y = pd.Series(np.random.binomial(1, 0.5, 250))

        with pytest.raises(ValueError):
            target_binary_feature_real_test(x=x, y=y, test="other_unknown_function")

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
