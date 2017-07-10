# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import numpy as np
import pandas as pd
from pytest import raises
from future.utils import lrange

from tsfresh.feature_selection.selection import select_features


def test_assert_list():
    with raises(TypeError):
        select_features(pd.DataFrame(index=range(2)),[1,2,3])


def test_assert_one_row_X():
    X = pd.DataFrame([1], index=[1])
    y = pd.Series([1], index=[1])
    with raises(ValueError):
        select_features(X, y)


def test_assert_different_index():
    X = pd.DataFrame(list(range(3)), index=[1, 2, 3])
    y = pd.Series(range(3), index=[1, 3, 4])
    with raises(ValueError):
        select_features(X, y)


def test_assert_shorter_y():
    X = pd.DataFrame([1, 2], index=[1, 2])
    y = np.array([1])
    with raises(ValueError):
        select_features(X, y)


def test_expects_more_than_two_columns_for_multiclass():
    X = pd.DataFrame([1, 2], index=[1, 2])
    y = np.array([1, 2])
    with raises(RuntimeError):
        select_features(X, y, multiclass=True)


def test_selects_for_each_class():
    df = pd.DataFrame()
    df['f1'] = [10] * 10 + lrange(10) + lrange(10)
    df['f2'] = lrange(10) + [10] * 10 + lrange(10)
    df['f3'] = lrange(10) + lrange(10) + [1] * 10
    df['y'] = [0] * 10 + [10] * 10 + [2] * 10

    y = df.y
    X = df.drop(['y'], axis=1)
    X_relevant = select_features(X, y, multiclass=True)
    assert {'f1', 'f2', 'f3'} == set(X_relevant.columns)
