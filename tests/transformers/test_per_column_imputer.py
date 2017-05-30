# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from builtins import range
from unittest import TestCase
import pandas as pd
import pandas.util.testing as pdt

import numpy as np
import numpy.testing as npt

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

        X_numpy = X.values

        imputer.fit(X)
        selected_X = imputer.transform(X)

        #re-initialize for new dicts
        imputer = PerColumnImputer()
        imputer.fit(X_numpy)
        selected_X_numpy = imputer.transform(X_numpy)

        npt.assert_array_equal(selected_X.values, selected_X_numpy.values)

        self.assertTrue(selected_X_numpy.shape, (1, 100))

    def test_standard_replacement_behavior(self):
        imputer = PerColumnImputer()

        data = [np.NINF, np.PINF, np.nan, 100.0, -100.0, 1.0, 1.0]
        truth = [-100.0, 100.0, 1.0, 100.0, -100.0, 1.0, 1.0]
        X = pd.DataFrame({"a": data})
        true_X = pd.DataFrame({"a": truth})

        imputer.fit(X)
        selected_X = imputer.transform(X)

        pdt.assert_frame_equal(selected_X, true_X)

    def test_only_NINF_repl_given(self):
        data = [np.NINF, np.PINF, np.nan, 100.0, -100.0, 1.0, 1.0]
        truth = [-100.0, 100.0, 1.0, 100.0, -100.0, 1.0, 1.0]
        X = pd.DataFrame({"a": data})
        true_X = pd.DataFrame({"a": truth})

        col_to_min = {"a": -100}
        imputer = PerColumnImputer(col_to_NINF_repl=col_to_min)

        imputer.fit(X)
        selected_X = imputer.transform(X)

        pdt.assert_frame_equal(selected_X, true_X)

    def test_only_PINF_repl_given(self):
        data = [np.NINF, np.PINF, np.nan, 100.0, -100.0, 1.0, 1.0]
        truth = [-100.0, 100.0, 1.0, 100.0, -100.0, 1.0, 1.0]
        X = pd.DataFrame({"a": data})
        true_X = pd.DataFrame({"a": truth})

        col_to_max = {"a": 100}
        imputer = PerColumnImputer(col_to_PINF_repl=col_to_max)

        imputer.fit(X)
        selected_X = imputer.transform(X)

        pdt.assert_frame_equal(selected_X, true_X)

    def test_only_NAN_repl_given(self):
        data = [np.NINF, np.PINF, np.nan, 100.0, -100.0, 1.0, 1.0]
        truth = [-100.0, 100.0, 1.0, 100.0, -100.0, 1.0, 1.0]
        X = pd.DataFrame({"a": data})
        true_X = pd.DataFrame({"a": truth})

        col_to_median = {"a": 1}
        imputer = PerColumnImputer(col_to_NAN_repl=col_to_median)

        imputer.fit(X)
        selected_X = imputer.transform(X)

        pdt.assert_frame_equal(selected_X, true_X)