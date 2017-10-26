# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from __future__ import absolute_import, division

from unittest import TestCase
import numpy as np
import pandas as pd
from tsfresh.feature_extraction.extraction import extract_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, MinimalFCParameters, \
    EfficientFCParameters, from_columns
import six
from tsfresh.feature_extraction import feature_calculators
from pandas.testing import assert_frame_equal


class TestSettingsObject(TestCase):
    """
    This tests the base class ComprehensiveFCParameters
    """

    def test_from_columns(self):
        tsn = "TEST_TIME_SERIES"

        fset = ComprehensiveFCParameters()
        self.assertRaises(TypeError, from_columns, 42)
        self.assertRaises(TypeError, from_columns, 42)
        self.assertRaises(ValueError, from_columns, ["This is not a column name"])
        self.assertRaises(ValueError, from_columns, ["This__neither"])
        self.assertRaises(ValueError, from_columns, ["This__also__not"])

        # Aggregate functions
        feature_names = [tsn + '__sum_values', tsn + "__median", tsn + "__length", tsn + "__sample_entropy"]

        # Aggregate functions with params
        feature_names += [tsn + '__quantile__q_10', tsn + '__quantile__q_70', tsn + '__number_peaks__n_30',
                          tsn + '__value_count__value_inf', tsn + '__value_count__value_-inf',
                          tsn + '__value_count__value_nan']

        # Apply functions
        feature_names += [tsn + '__ar_coefficient__k_20__coeff_4', tsn + '__ar_coefficient__coeff_10__k_-1']

        kind_to_fc_parameters = from_columns(feature_names)

        six.assertCountEqual(self, list(kind_to_fc_parameters[tsn].keys()),
                             ["sum_values", "median", "length", "sample_entropy", "quantile", "number_peaks",
                              "ar_coefficient", "value_count"])

        self.assertEqual(kind_to_fc_parameters[tsn]["sum_values"], None)
        self.assertEqual(kind_to_fc_parameters[tsn]["ar_coefficient"],
                         [{"k": 20, "coeff": 4}, {"k": -1, "coeff": 10}])

        self.assertEqual(kind_to_fc_parameters[tsn]["value_count"],
                         [{"value": np.PINF}, {"value": np.NINF}, {"value": np.NaN}])

        # test that it passes for all functions
        fset = ComprehensiveFCParameters()
        X_org = extract_features(pd.DataFrame({"value": [1, 2, 3], "id": [1, 1, 1]}),
                                 default_fc_parameters=fset,
                                 column_id="id", column_value="value",
                                 n_jobs=0)

        inferred_fset = from_columns(X_org)

        X_new = extract_features(pd.DataFrame({"value": [1, 2, 3], "id": [1, 1, 1]}),
                                 kind_to_fc_parameters=inferred_fset,
                                 column_id="id", column_value="value",
                                 n_jobs=0)

        assert_frame_equal(X_org.sort_index(), X_new.sort_index())

    def test_from_columns_ignores_columns(self):

        tsn = "TEST_TIME_SERIES"
        feature_names = [tsn + '__sum_values', tsn + "__median", tsn + "__length", tsn + "__sample_entropy"]
        feature_names += ["THIS_COL_SHOULD_BE_IGNORED"]

        kind_to_fc_parameters = from_columns(feature_names, columns_to_ignore=["THIS_COL_SHOULD_BE_IGNORED",
                                                                               "THIS_AS_WELL"])

        six.assertCountEqual(self, list(kind_to_fc_parameters[tsn].keys()),
                             ["sum_values", "median", "length", "sample_entropy"])

    def test_default_calculates_all_features(self):
        """
        Test that by default a ComprehensiveFCParameters object should be set up to calculate all features defined
        in tsfresh.feature_extraction.feature_calculators
        """
        settings = ComprehensiveFCParameters()
        all_feature_calculators = [name for name, func in feature_calculators.__dict__.items()
                                   if hasattr(func, "fctype")]

        for calculator in all_feature_calculators:
            self.assertIn(calculator, settings,
                          msg='Default ComprehensiveFCParameters object does not setup calculation of {}'
                          .format(calculator))


class TestEfficientFCParameters(TestCase):
    """
    This tests the EfficientFCParameters( class
    """

    def test_extraction_runs_through(self):
        rfs = EfficientFCParameters()
        data = pd.DataFrame([[0, 0, 0, 0], [1, 0, 0, 0]], columns=["id", "time", "kind", "value"])

        extracted_features = extract_features(data, default_fc_parameters=rfs,
                                              column_kind="kind", column_value="value",
                                              column_sort="time", column_id="id")

        six.assertCountEqual(self, extracted_features.index, [0, 1])

    def test_contains_all_non_high_comp_cost_features(self):
        """
        Test that by default a EfficientFCParameters object should be set up to calculate all features defined
        in tsfresh.feature_extraction.feature_calculators that do not have the attribute "high_comp_cost"
        """
        rfs = EfficientFCParameters()
        all_feature_calculators = [name for name, func in feature_calculators.__dict__.items()
                                   if hasattr(func, "fctype") and not hasattr(func, "high_comp_cost")]

        for calculator in all_feature_calculators:
            self.assertIn(calculator, rfs,
                          msg='Default EfficientFCParameters object does not setup calculation of {}'
                          .format(calculator))


class TestMinimalSettingsObject(TestCase):
    def test_all_minimal_features_in(self):
        mfs = MinimalFCParameters()

        self.assertIn("mean", mfs)
        self.assertIn("median", mfs)
        self.assertIn("minimum", mfs)
        self.assertIn("maximum", mfs)
        self.assertIn("length", mfs)
        self.assertIn("sum_values", mfs)
        self.assertIn("standard_deviation", mfs)
        self.assertIn("variance", mfs)
        self.assertNotIn("fft_coefficient", mfs)

    def test_extraction_runs_through(self):
        mfs = MinimalFCParameters()

        data = pd.DataFrame([[0, 0, 0, 0], [1, 0, 0, 0]], columns=["id", "time", "kind", "value"])

        extracted_features = extract_features(data, default_fc_parameters=mfs,
                                              column_kind="kind", column_value="value",
                                              column_sort="time", column_id="id")

        six.assertCountEqual(self, extracted_features.columns, ["0__median", "0__standard_deviation", "0__sum_values",
                                                                "0__maximum", "0__variance", "0__minimum", "0__mean",
                                                                "0__length"])
        six.assertCountEqual(self, extracted_features.index, [0, 1])