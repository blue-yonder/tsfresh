# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from builtins import range
from unittest import TestCase
import pandas as pd

import numpy as np

from tsfresh.transformers.feature_selector import FeatureSelector

class FeatureSelectorTestCase(TestCase):
    def setUp(self):
        np.random.seed(0)


    def test_not_fitted(self):
        selector = FeatureSelector()

        X = pd.DataFrame()

        self.assertRaises(RuntimeError, selector.transform, X)

    def test_extract_relevant_features(self):
        selector = FeatureSelector()
        np.random.seed(42)
        y = pd.Series(np.random.binomial(1, 0.5, 1000))
        X = pd.DataFrame(index=list(range(1000)))

        z = y - np.random.binomial(1, 0.1, 1000) + np.random.binomial(1, 0.1, 1000)
        z[z == -1] = 0
        z[z == 2] = 1

        X["rel1"] = z
        X["rel2"] = y * np.abs(np.random.normal(0, 1, 1000)) + np.random.normal(0, 0.1, 1000)
        X["rel3"] = y + np.random.normal(0, 1, 1000)
        X["rel4"] = y ** 2 + np.random.normal(0, 1, 1000)
        X["rel5"] = np.sqrt(y) + np.random.binomial(2, 0.1, 1000)

        X["irr_constant"] = 1.113344

        X["irr1"] = np.random.normal(0, 1, 1000)
        X["irr2"] = np.random.poisson(1, 1000)
        X["irr3"] = np.random.binomial(1, 0.3, 1000)
        X["irr4"] = np.random.normal(0, 1, 1000)
        X["irr5"] = np.random.poisson(1, 1000)
        X["irr6"] = np.random.binomial(1, 0.3, 1000)
        X["irr7"] = np.random.normal(0, 1, 1000)
        X["irr8"] = np.random.poisson(1, 1000)
        X["irr9"] = np.random.binomial(1, 0.3, 1000)

        returned_selector = selector.fit(X, y)
        self.assertIs(returned_selector, selector)

        self.assertEqual(sorted(list(selector.relevant_features.index)), ["rel1", "rel2", "rel3", "rel4", "rel5"])

        new_X = X.copy()

        selected_X = selector.transform(new_X)

        self.assertEqual(sorted(list(selector.relevant_features.index)), sorted(list(selected_X.columns)))

    def test_nothing_relevant(self):
        selector = FeatureSelector()

        y = pd.Series(np.random.binomial(1, 0.5, 1000))
        X = pd.DataFrame(index=list(range(1000)))

        X["irr1"] = np.random.normal(0, 1, 1000)
        X["irr2"] = np.random.normal(2, 1, 1000)

        selector.fit(X, y)

        transformed_X = selector.transform(X.copy())

        self.assertEqual(list(transformed_X.columns), [])
        self.assertEqual(list(transformed_X.index), list(X.index))

    def test_with_numpy_array(self):
        selector = FeatureSelector()

        y = pd.Series(np.random.binomial(1, 0.5, 1000))
        X = pd.DataFrame(index=list(range(1000)))

        X["irr1"] = np.random.normal(0, 1, 1000)
        X["rel1"] = y

        y_numpy = y.values
        X_numpy = X.as_matrix()

        selector.fit(X, y)
        selected_X = selector.transform(X)

        selector.fit(X_numpy, y_numpy)
        selected_X_numpy = selector.transform(X_numpy)

        self.assertTrue((selected_X_numpy == selected_X.values).all())

        self.assertTrue(selected_X_numpy.shape, (1, 1000))



