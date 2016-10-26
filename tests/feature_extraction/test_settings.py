# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from __future__ import absolute_import, division

from unittest import TestCase

import numpy as np

from tsfresh.feature_extraction.settings import FeatureExtractionSettings

class TestSettingsObject(TestCase):

    def test_from_columns(self):

        tsn = "TEST_TIME_SERIES"

        fset = FeatureExtractionSettings()
        self.assertRaises(TypeError, fset.from_columns, 42)
        self.assertRaises(TypeError, fset.from_columns, 42)
        self.assertRaises(ValueError, fset.from_columns, ["This is not a column name"])
        self.assertRaises(ValueError, fset.from_columns, ["This__neither"])
        self.assertRaises(ValueError, fset.from_columns, ["This__also__not"])

        # Aggregate functions
        feature_names = [tsn + '__sum_values', tsn + "__median", tsn + "__length"]

        # Aggregate functions with params
        feature_names += [tsn + '__quantile__q_10', tsn + '__quantile__q_70', tsn + '__number_peaks__n_30',
                          tsn + '__value_count__value_inf', tsn + '__value_count__value_-inf',
                          tsn + '__value_count__value_nan']

        # Apply functions
        feature_names += [tsn + '__ar_coefficient__k_20__coeff_4', tsn + '__ar_coefficient__coeff_10__k_-1']

        cset = fset.from_columns(feature_names)

        self.assertItemsEqual(cset.kind_to_calculation_settings_mapping[tsn].keys(),
                              ["sum_values", "median", "length", "quantile", "number_peaks", "ar_coefficient",
                               "value_count"])

        self.assertEqual(cset.kind_to_calculation_settings_mapping[tsn]["sum_values"], None)
        self.assertEqual(cset.kind_to_calculation_settings_mapping[tsn]["ar_coefficient"],
                         [{"k": 20, "coeff": 4}, {"k": -1, "coeff": 10}])

        self.assertEqual(cset.kind_to_calculation_settings_mapping[tsn]["value_count"],
                         [{"value": np.PINF}, {"value": np.NINF}, {"value": np.NaN}])