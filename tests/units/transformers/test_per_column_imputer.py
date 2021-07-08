# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import warnings
from builtins import range
from unittest import TestCase

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
from sklearn.exceptions import NotFittedError

from tsfresh.transformers.per_column_imputer import PerColumnImputer


class PerColumnImputerTestCase(TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_not_fitted(self):
        imputer = PerColumnImputer()

        X = pd.DataFrame()

        self.assertRaises(NotFittedError, imputer.transform, X)

    def test_only_nans_and_infs(self):
        imputer = PerColumnImputer()

        X = pd.DataFrame(index=list(range(100)))

        X["NaNs"] = np.nan * np.ones(100)
        X["PINF"] = np.PINF * np.ones(100)
        X["NINF"] = np.NINF * np.ones(100)

        with warnings.catch_warnings(record=True) as w:
            imputer.fit(X)
            self.assertEqual(len(w), 1)
            self.assertEqual(
                "The columns ['NaNs' 'PINF' 'NINF'] did not have any finite values. Filling with zeros.",
                str(w[0].message),
            )

        selected_X = imputer.transform(X)

        self.assertTrue((selected_X.values == 0).all())

    def test_with_numpy_array(self):
        imputer = PerColumnImputer()

        X = pd.DataFrame(index=list(range(100)))

        X["NaNs"] = np.nan * np.ones(100)
        X["PINF"] = np.PINF * np.ones(100)
        X["NINF"] = np.NINF * np.ones(100)

        X_numpy = X.values.copy()

        with warnings.catch_warnings(record=True) as w:
            imputer.fit(X)
            self.assertEqual(len(w), 1)
            self.assertEqual(
                "The columns ['NaNs' 'PINF' 'NINF'] did not have any finite values. Filling with zeros.",
                str(w[0].message),
            )

        selected_X = imputer.transform(X)

        # re-initialize for new dicts
        imputer = PerColumnImputer()
        with warnings.catch_warnings(record=True) as w:
            imputer.fit(X_numpy)
            self.assertEqual(len(w), 1)
            self.assertEqual(
                "The columns [0 1 2] did not have any finite values. Filling with zeros.",
                str(w[0].message),
            )

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

    def test_partial_preset_col_to_NINF_given(self):
        data = [np.NINF, np.PINF, np.nan, 100.0, -100.0, 1.0, 1.0]
        truth = [-100.0, 100.0, 1.0, 100.0, -100.0, 1.0, 1.0]
        X = pd.DataFrame({"a": data})
        true_X = pd.DataFrame({"a": truth})

        col_to_min = {"a": -100}
        imputer = PerColumnImputer(col_to_NINF_repl_preset=col_to_min)

        imputer.fit(X)
        selected_X = imputer.transform(X)

        pdt.assert_frame_equal(selected_X, true_X)

    def test_partial_preset_col_to_PINF_given(self):
        data = [np.NINF, np.PINF, np.nan, 100.0, -100.0, 1.0, 1.0]
        truth = [-100.0, 100.0, 1.0, 100.0, -100.0, 1.0, 1.0]
        X = pd.DataFrame({"a": data})
        true_X = pd.DataFrame({"a": truth})

        col_to_max = {"a": 100}
        imputer = PerColumnImputer(col_to_PINF_repl_preset=col_to_max)

        imputer.fit(X)
        selected_X = imputer.transform(X)

        pdt.assert_frame_equal(selected_X, true_X)

    def test_partial_preset_col_to_NAN_given(self):
        data = [np.NINF, np.PINF, np.nan, 100.0, -100.0, 1.0, 1.0]
        truth = [-100.0, 100.0, 1.0, 100.0, -100.0, 1.0, 1.0]
        X = pd.DataFrame({"a": data})
        true_X = pd.DataFrame({"a": truth})

        col_to_median = {"a": 1}
        imputer = PerColumnImputer(col_to_NAN_repl_preset=col_to_median)

        imputer.fit(X)
        selected_X = imputer.transform(X)

        pdt.assert_frame_equal(selected_X, true_X)

    def test_different_shapes_fitted_and_transformed(self):
        imputer = PerColumnImputer()

        X = pd.DataFrame(index=list(range(10)))
        X["a"] = np.ones(10)

        imputer.fit(X)
        X["b"] = np.ones(10)

        self.assertRaises(ValueError, imputer.transform, X)

    def test_preset_has_higher_priority_than_fit(self):
        data = [np.NINF, np.PINF, np.nan, 100.0, -100.0, 1.0, 1.0]
        truth = [-100.0, 100.0, 0.0, 100.0, -100.0, 1.0, 1.0]

        X = pd.DataFrame({"a": data})
        true_X = pd.DataFrame({"a": truth})

        col_to_median = {"a": 0}
        imputer = PerColumnImputer(col_to_NAN_repl_preset=col_to_median)
        imputer.fit(X)

        selected_X = imputer.transform(X)

        pdt.assert_frame_equal(selected_X, true_X)

    def test_only_parameters_of_last_fit_count(self):
        data = [np.NINF, np.PINF, np.nan, 100.0, -100.0, 1.0, 1.0]
        data_2 = [np.NINF, np.PINF, np.nan, 10.0, -10.0, 3.0, 3.0]
        truth_a = [-10.0, 10.0, 3.0, 10.0, -10.0, 3.0, 3.0]
        truth_b = [-10.0, 10.0, 3.0, 10.0, -10.0, 3.0, 3.0]

        X = pd.DataFrame({"a": data, "b": data})
        X_2 = pd.DataFrame({"a": data_2, "b": data_2})
        true_X = pd.DataFrame({"a": truth_a, "b": truth_b})

        imputer = PerColumnImputer()

        imputer.fit(X)
        imputer.fit(X_2)

        selected_X = imputer.transform(X_2)

        pdt.assert_frame_equal(selected_X, true_X)

    def test_only_subset_of_columns_given(self):
        data = [np.NINF, np.PINF, np.nan, 100.0, -100.0, 1.0, 1.0]
        truth_a = [-100.0, 100.0, 0.0, 100.0, -100.0, 1.0, 1.0]
        truth_b = [-100.0, 100.0, 1.0, 100.0, -100.0, 1.0, 1.0]
        X = pd.DataFrame({"a": data, "b": data})
        true_X = pd.DataFrame({"a": truth_a, "b": truth_b})

        col_to_median = {"a": 0}
        imputer = PerColumnImputer(col_to_NAN_repl_preset=col_to_median)

        imputer.fit(X)
        selected_X = imputer.transform(X)

        pdt.assert_frame_equal(selected_X, true_X)

    def test_NINF_preset_contains_more_columns_than_dataframe_to_fit(self):
        X = pd.DataFrame(index=list(range(10)))
        X["a"] = np.ones(10)

        col_to_min = {"a": 0, "b": 0}

        imputer = PerColumnImputer(col_to_NINF_repl_preset=col_to_min)

        self.assertRaises(ValueError, imputer.fit, X)

    def test_PINF_preset_contains_more_columns_than_dataframe_to_fit(self):
        X = pd.DataFrame(index=list(range(10)))
        X["a"] = np.ones(10)

        col_to_max = {"a": 0, "b": 0}

        imputer = PerColumnImputer(col_to_PINF_repl_preset=col_to_max)

        self.assertRaises(ValueError, imputer.fit, X)

    def test_NAN_preset_contains_more_columns_than_dataframe_to_fit(self):
        X = pd.DataFrame(index=list(range(10)))
        X["a"] = np.ones(10)

        col_to_median = {"a": 0, "b": 0}

        imputer = PerColumnImputer(col_to_NAN_repl_preset=col_to_median)

        self.assertRaises(ValueError, imputer.fit, X)
