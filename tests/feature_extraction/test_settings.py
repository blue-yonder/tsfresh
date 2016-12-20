# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from __future__ import absolute_import, division

from unittest import TestCase
import numpy as np
import pandas as pd
from tsfresh.feature_extraction.extraction import extract_features, _extract_features_for_one_time_series
from tsfresh.feature_extraction.settings import FeatureExtractionSettings, MinimalFeatureExtractionSettings,\
    ReasonableFeatureExtractionSettings
import six
from tsfresh.feature_extraction import feature_calculators


class TestSettingsObject(TestCase):
    """
    This tests the base class FeatureExtractionSettings
    """
    def test_from_columns(self):
        tsn = "TEST_TIME_SERIES"

        fset = FeatureExtractionSettings()
        self.assertRaises(TypeError, fset.from_columns, 42)
        self.assertRaises(TypeError, fset.from_columns, 42)
        self.assertRaises(ValueError, fset.from_columns, ["This is not a column name"])
        self.assertRaises(ValueError, fset.from_columns, ["This__neither"])
        self.assertRaises(ValueError, fset.from_columns, ["This__also__not"])

        # Aggregate functions
        feature_names = [tsn + '__sum_values', tsn + "__median", tsn + "__length", tsn + "__sample_entropy"]


        # Aggregate functions with params
        feature_names += [tsn + '__quantile__q_10', tsn + '__quantile__q_70', tsn + '__number_peaks__n_30',
                          tsn + '__value_count__value_inf', tsn + '__value_count__value_-inf',
                          tsn + '__value_count__value_nan']

        # Apply functions
        feature_names += [tsn + '__ar_coefficient__k_20__coeff_4', tsn + '__ar_coefficient__coeff_10__k_-1']

        cset = fset.from_columns(feature_names)

        six.assertCountEqual(self, list(cset.kind_to_calculation_settings_mapping[tsn].keys()), 
          ["sum_values", "median", "length", "sample_entropy", "quantile", "number_peaks", "ar_coefficient",
                                  "value_count"])
        
        self.assertEqual(cset.kind_to_calculation_settings_mapping[tsn]["sum_values"], None)
        self.assertEqual(cset.kind_to_calculation_settings_mapping[tsn]["ar_coefficient"],
                         [{"k": 20, "coeff": 4}, {"k": -1, "coeff": 10}])

        self.assertEqual(cset.kind_to_calculation_settings_mapping[tsn]["value_count"],
                         [{"value": np.PINF}, {"value": np.NINF}, {"value": np.NaN}])

    def test_default_calculates_all_features(self):
        """
        Test that by default a FeatureExtractionSettings object should be set up to calculate all features defined
        in tsfresh.feature_extraction.feature_calculators
        """
        settings = FeatureExtractionSettings()
        all_feature_calculators = [name for name, func in feature_calculators.__dict__.items()
                                   if hasattr(func, "fctype")]

        for calculator in all_feature_calculators:
            self.assertIn(calculator, settings.name_to_param,
                          msg='Default FeatureExtractionSettings object does not setup calculation of {}'
                          .format(calculator))


class TestReasonableFeatureExtractionSettings(TestCase):
    """
    This tests the ReasonableFeatureExtractionSettings class
    """

    def test_extraction_runs_through(self):
        rfs = ReasonableFeatureExtractionSettings()

        data = pd.DataFrame([[0, 0, 0, 0], [1, 0, 0, 0]], columns=["id", "time", "kind", "value"])

        extracted_features = extract_features(data, feature_extraction_settings=rfs,
                                              column_kind="kind", column_value="value",
                                              column_sort="time", column_id="id")

        six.assertCountEqual(self, extracted_features.index, [0, 1])

    def test_contains_all_non_high_comp_cost_features(self):
        """
        Test that by default a FeatureExtractionSettings object should be set up to calculate all features defined
        in tsfresh.feature_extraction.feature_calculators that do not have the attribute "high_comp_cost"
        """
        rfs = ReasonableFeatureExtractionSettings()
        all_feature_calculators = [name for name, func in feature_calculators.__dict__.items()
                                   if hasattr(func, "fctype") and not hasattr(func, "high_comp_cost")]

        for calculator in all_feature_calculators:
            self.assertIn(calculator, rfs.name_to_param,
                          msg='Default FeatureExtractionSettings object does not setup calculation of {}'
                          .format(calculator))


class TestMinimalSettingsObject(TestCase):
    def test_all_minimal_features_in(self):
        mfs = MinimalFeatureExtractionSettings()

        self.assertIn("mean", mfs.name_to_param)
        self.assertIn("median", mfs.name_to_param)
        self.assertIn("minimum", mfs.name_to_param)
        self.assertIn("maximum", mfs.name_to_param)
        self.assertIn("length", mfs.name_to_param)
        self.assertIn("sum_values", mfs.name_to_param)
        self.assertIn("standard_deviation", mfs.name_to_param)
        self.assertIn("variance", mfs.name_to_param)

    def test_extraction_runs_through(self):
        mfs = MinimalFeatureExtractionSettings()

        data = pd.DataFrame([[0, 0, 0, 0], [1, 0, 0, 0]], columns=["id", "time", "kind", "value"])

        extracted_features = extract_features(data, feature_extraction_settings=mfs,
                                              column_kind="kind", column_value="value",
                                              column_sort="time", column_id="id")

        six.assertCountEqual(self, extracted_features.columns, ["0__median", "0__standard_deviation", "0__sum_values",
                                                                "0__maximum", "0__variance","0__minimum", "0__mean",
                                                                "0__length"])
        six.assertCountEqual(self, extracted_features.index, [0, 1])

    def test_extraction_for_one_time_series_runs_through(self):
        mfs = MinimalFeatureExtractionSettings()
        data = pd.DataFrame([[0, 0, 0, 0], [1, 0, 0, 0]], columns=["id", "time", "kind", "value"])
        extracted_features = _extract_features_for_one_time_series([0, data], settings=mfs,
                                                                   column_value="value", column_id="id")
        six.assertCountEqual(self, extracted_features.columns,
                             ["0__median", "0__standard_deviation", "0__sum_values", "0__maximum", "0__variance",
                              "0__minimum", "0__mean", "0__length"])
        six.assertCountEqual(self, extracted_features.index, [0, 1])