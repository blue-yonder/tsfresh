# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from builtins import range
from unittest import TestCase
import pandas as pd

import numpy as np

from tsfresh.transformers.per_column_imputer import PerColumnImputer


class PerColumnImputerTestCase(TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_not_fitted(self):
        imputer = PerColumnImputer()

        X = pd.DataFrame()

        self.assertRaises(RuntimeError, imputer.transform, X)

    def test_only_nans_and_infs(self):
        imputer = PerColumnImputer()

        X = pd.DataFrame(index=list(range(100)))

        X["NaNs"] = np.nan * np.ones(100)
        X["PINF"] = np.PINF * np.ones(100)
        X["NINF"] = np.NINF * np.ones(100)

        imputer.fit(X)
        selected_X = imputer.transform(X)

        self.assertTrue((selected_X.values == 0).all())

    def test_with_numpy_array(self):
        imputer = PerColumnImputer()

        X = pd.DataFrame(index=list(range(100)))

        X["NaNs"] = np.nan * np.ones(100)
        X["PINF"] = np.PINF * np.ones(100)
        X["NINF"] = np.NINF * np.ones(100)

        X_numpy = X.as_matrix()

        imputer.fit(X)
        selected_X = imputer.transform(X)

        #re-initialize for new dicts
        imputer = PerColumnImputer()
        imputer.fit(X_numpy)
        selected_X_numpy = imputer.transform(X_numpy)

        self.assertTrue((selected_X_numpy == selected_X.values).all().all())

        self.assertTrue(selected_X_numpy.shape, (1, 100))

    def test_standard_replacement_behavior(self):
        imputer = PerColumnImputer()

        X = pd.DataFrame(index=list(range(10)))
        X["a"] = np.ones(10)
        X["a"][3] = 100
        X["a"][4] = -100
        true_X = X
        true_X["a"][0] = -100
        true_X["a"][1] = 100

        X["a"][0] = np.NINF
        X["a"][1] = np.PINF
        X["a"][2] = np.nan

        imputer.fit(X)
        selected_X = imputer.transform(X)

        self.assertTrue((selected_X.values == true_X.values).all().all())

    def test_only_NINF_repl_given(self):
        X = pd.DataFrame(index=list(range(10)))
        X["a"] = np.ones(10)
        X["a"][3] = 100
        X["a"][4] = -100
        true_X = X
        true_X["a"][0] = -100
        true_X["a"][1] = 100

        X["a"][0] = np.NINF
        X["a"][1] = np.PINF
        X["a"][2] = np.nan

        col_to_min = {"a": -100}
        imputer = PerColumnImputer(col_to_NINF_repl=col_to_min)

        imputer.fit(X)
        selected_X = imputer.transform(X)

        self.assertTrue((selected_X.values == true_X.values).all().all())

    def test_only_PINF_repl_given(self):
        X = pd.DataFrame(index=list(range(10)))
        X["a"] = np.ones(10)
        X["a"][3] = 100
        X["a"][4] = -100
        true_X = X
        true_X["a"][0] = -100
        true_X["a"][1] = 100

        X["a"][0] = np.NINF
        X["a"][1] = np.PINF
        X["a"][2] = np.nan

        col_to_max = {"a": 100}
        imputer = PerColumnImputer(col_to_PINF_repl=col_to_max)

        imputer.fit(X)
        selected_X = imputer.transform(X)

        self.assertTrue((selected_X.values == true_X.values).all().all())

    def test_only_NAN_repl_given(self):
        X = pd.DataFrame(index=list(range(10)))
        X["a"] = np.ones(10)
        X["a"][3] = 100
        X["a"][4] = -100
        true_X = X
        true_X["a"][0] = -100
        true_X["a"][1] = 100

        X["a"][0] = np.NINF
        X["a"][1] = np.PINF
        X["a"][2] = np.nan

        col_to_median = {"a": 1}
        imputer = PerColumnImputer(col_to_NAN_repl=col_to_median)

        imputer.fit(X)
        selected_X = imputer.transform(X)

        self.assertTrue((selected_X.values == true_X.values).all().all())