# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import numpy as np
import pandas as pd
import pytest
from future.utils import lrange

from tsfresh.feature_selection.selection import select_features


class TestSelectFeatures:
    def test_assert_list(self):
        with pytest.raises(TypeError):
            select_features(pd.DataFrame(index=range(2)),[1,2,3])


    def test_assert_one_row_X(self):
        X = pd.DataFrame([1], index=[1])
        y = pd.Series([1], index=[1])
        with pytest.raises(ValueError):
            select_features(X, y)

    def test_assert_one_label_y(self):
        X = pd.DataFrame([10, 10], index=[1, 2])
        y = pd.Series([1, 1], index=[1, 2])
        with pytest.raises(ValueError):
            select_features(X, y)

    def test_assert_different_index(self):
        X = pd.DataFrame(list(range(3)), index=[1, 2, 3])
        y = pd.Series(range(3), index=[1, 3, 4])
        with pytest.raises(ValueError):
            select_features(X, y)


    def test_assert_shorter_y(self):
        X = pd.DataFrame([1, 2], index=[1, 2])
        y = np.array([1])
        with pytest.raises(ValueError):
            select_features(X, y)


    def test_selects_for_each_class(self):
        df = pd.DataFrame()
        df['f1'] = [10] * 10 + lrange(10) + lrange(10)
        df['f2'] = lrange(10) + [10] * 10 + lrange(10)
        df['f3'] = lrange(10) + lrange(10) + [10] * 10
        df['y'] = [0] * 10 + [1] * 10 + [2] * 10

        y = df.y
        X = df.drop(['y'], axis=1)
        X_relevant = select_features(X, y, ml_task='classification')
        assert {'f1', 'f2', 'f3'} == set(X_relevant.columns)
