# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import numpy as np
from unittest import TestCase
import pandas as pd

from tsfresh.feature_selection.selection import select_features


class SelectionTestCase(TestCase):

    def test_assert_list(self):
        self.assertRaises(TypeError, select_features, X=pd.DataFrame(index=range(2)), y=[1,2,3])

    def test_assert_one_row_X(self):
        X = pd.DataFrame([1], index=[1])
        y = pd.Series([1], index=[1])
        self.assertRaises(ValueError, select_features, X=X, y=y)

    def test_assert_different_index(self):
        X = pd.DataFrame(list(range(3)), index=[1, 2, 3])
        y = pd.Series(range(3), index=[1, 3, 4])
        self.assertRaises(ValueError, select_features, X=X, y=y)

    def test_assert_shorter_y(self):
        X = pd.DataFrame([1, 2], index=[1, 2])
        y = np.array([1])
        self.assertRaises(ValueError, select_features, X=X, y=y)
