# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import numpy as np
import pandas as pd
import pytest

from tsfresh.feature_selection.selection import select_features


class TestSelectFeatures:
    def test_assert_list(self):
        with pytest.raises(AssertionError):
            select_features(pd.DataFrame(index=range(2)), [1, 2, 3])

    def test_assert_one_row_X(self):
        X = pd.DataFrame([1], index=[1])
        y = pd.Series([1], index=[1])
        with pytest.raises(AssertionError):
            select_features(X, y)

    def test_assert_one_label_y(self):
        X = pd.DataFrame([10, 10], index=[1, 2])
        y = pd.Series([1, 1], index=[1, 2])
        with pytest.raises(AssertionError):
            select_features(X, y)

    def test_assert_different_index(self):
        X = pd.DataFrame(list(range(3)), index=[1, 2, 3])
        y = pd.Series(range(3), index=[1, 3, 4])
        with pytest.raises(ValueError):
            select_features(X, y)

    def test_assert_shorter_y(self):
        X = pd.DataFrame([1, 2], index=[1, 2])
        y = np.array([1])
        with pytest.raises(AssertionError):
            select_features(X, y)

    def test_assert_X_is_DataFrame(self):
        X = np.array([[1, 2], [1, 2]])
        y = np.array([1])
        with pytest.raises(AssertionError):
            select_features(X, y)

    def test_selects_for_each_class(self):
        df = pd.DataFrame()
        df["f1"] = [10] * 10 + list(range(10)) + list(range(10))
        df["f2"] = list(range(10)) + [10] * 10 + list(range(10))
        df["f3"] = list(range(10)) + list(range(10)) + [10] * 10
        df["y"] = [0] * 10 + [1] * 10 + [2] * 10

        y = df.y
        X = df.drop(["y"], axis=1)
        X_relevant = select_features(X, y, ml_task="classification")
        assert {"f1", "f2", "f3"} == set(X_relevant.columns)

    def test_multiclass_selects_correct_n_significant(self):
        df = pd.DataFrame()
        N = 10
        constants = [N] * N
        increase = list(range(N))

        df["f1"] = constants + increase + increase
        df["f2"] = increase + constants + increase
        df["f3"] = increase + increase + constants
        df["y"] = [0] * N + [1] * N + [2] * N

        y = df.y
        X = df.drop(["y"], axis=1)
        X_relevant = select_features(
            X,
            y,
            ml_task="classification",
            multiclass=True,
            n_significant=1,
            fdr_level=0.01,
        )
        assert {"f1", "f2", "f3"} == set(X_relevant.columns)
        X_relevant = select_features(
            X,
            y,
            ml_task="classification",
            multiclass=True,
            n_significant=2,
            fdr_level=0.01,
        )
        assert len(X_relevant.columns) == 0
        X_relevant = select_features(
            X,
            y,
            ml_task="classification",
            multiclass=True,
            n_significant=3,
            fdr_level=0.01,
        )
        assert len(X_relevant.columns) == 0
