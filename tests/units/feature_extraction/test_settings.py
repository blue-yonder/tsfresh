# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import pickle
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from tsfresh.feature_extraction import feature_calculators
from tsfresh.feature_extraction.extraction import extract_features
from tsfresh.feature_extraction.settings import (
    ComprehensiveFCParameters,
    EfficientFCParameters,
    IndexBasedFCParameters,
    MinimalFCParameters,
    PickableSettings,
    TimeBasedFCParameters,
    from_columns,
)


class TestSettingsObject(TestCase):
    """
    This tests the base class ComprehensiveFCParameters
    """

    def test_range_count_correctly_configured(self):
        fset = ComprehensiveFCParameters()
        params_range_count = fset["range_count"]
        for param in params_range_count:
            assert param["min"] < param["max"]

    def test_from_column_raises_on_wrong_column_format(self):

        self.assertRaises(TypeError, from_columns, 42)
        self.assertRaises(TypeError, from_columns, [42])
        self.assertRaises(ValueError, from_columns, ["This is not a column name"])
        self.assertRaises(ValueError, from_columns, ["This__neither"])
        self.assertRaises(ValueError, from_columns, ["This__also__not"])

    def test_from_column_correct_for_selected_columns(self):
        tsn = "TEST_TIME_SERIES"

        # Aggregate functions
        feature_names = [
            tsn + "__sum_values",
            tsn + "__median",
            tsn + "__length",
            tsn + "__sample_entropy",
        ]

        # Aggregate functions with params
        feature_names += [
            tsn + "__quantile__q_10",
            tsn + "__quantile__q_70",
            tsn + "__number_peaks__n_30",
            tsn + "__value_count__value_inf",
            tsn + "__value_count__value_-inf",
            tsn + "__value_count__value_nan",
        ]

        # Apply functions
        feature_names += [
            tsn + "__ar_coefficient__k_20__coeff_4",
            tsn + "__ar_coefficient__coeff_10__k_-1",
        ]

        kind_to_fc_parameters = from_columns(feature_names)
        self.assertCountEqual(
            list(kind_to_fc_parameters[tsn].keys()),
            [
                "sum_values",
                "median",
                "length",
                "sample_entropy",
                "quantile",
                "number_peaks",
                "ar_coefficient",
                "value_count",
            ],
        )

        self.assertIsNone(kind_to_fc_parameters[tsn]["sum_values"])
        self.assertEqual(
            kind_to_fc_parameters[tsn]["ar_coefficient"],
            [{"k": 20, "coeff": 4}, {"k": -1, "coeff": 10}],
        )

        self.assertEqual(
            kind_to_fc_parameters[tsn]["value_count"],
            [{"value": np.PINF}, {"value": np.NINF}, {"value": np.NaN}],
        )

    def test_from_column_correct_for_comprehensive_fc_parameters(self):
        fset = ComprehensiveFCParameters()
        X_org = extract_features(
            pd.DataFrame({"value": [1, 2, 3], "id": [1, 1, 1]}),
            default_fc_parameters=fset,
            column_id="id",
            column_value="value",
            n_jobs=0,
        )
        inferred_fset = from_columns(X_org)
        X_new = extract_features(
            pd.DataFrame({"value": [1, 2, 3], "id": [1, 1, 1]}),
            kind_to_fc_parameters=inferred_fset,
            column_id="id",
            column_value="value",
            n_jobs=0,
        )
        assert_frame_equal(X_org.sort_index(), X_new.sort_index())

    def test_from_columns_ignores_columns(self):

        tsn = "TEST_TIME_SERIES"
        feature_names = [
            tsn + "__sum_values",
            tsn + "__median",
            tsn + "__length",
            tsn + "__sample_entropy",
        ]
        feature_names += ["THIS_COL_SHOULD_BE_IGNORED"]

        kind_to_fc_parameters = from_columns(
            feature_names,
            columns_to_ignore=["THIS_COL_SHOULD_BE_IGNORED", "THIS_AS_WELL"],
        )

        self.assertCountEqual(
            list(kind_to_fc_parameters[tsn].keys()),
            ["sum_values", "median", "length", "sample_entropy"],
        )

    def test_default_calculates_all_features(self):
        """
        Test that by default a ComprehensiveFCParameters object should be set up to calculate all features defined
        in tsfresh.feature_extraction.feature_calculators
        """
        settings = ComprehensiveFCParameters()
        all_feature_calculators = [
            name
            for name, func in feature_calculators.__dict__.items()
            if hasattr(func, "fctype") and not hasattr(func, "input_type")
        ]

        for calculator in all_feature_calculators:
            self.assertIn(
                calculator,
                settings,
                msg="Default ComprehensiveFCParameters object does not setup calculation of {}".format(
                    calculator
                ),
            )

    def test_from_columns_correct_for_different_kind_datatypes(self):
        """The `settings.from_columns()` function is supposed to save the feature extraction / selection results so it
        can be reused later. It works by parsing the column names of the extracted dataframes. An unfortunate side
        effect of this is that when used with the 'long' format time series input, the typing information about the
        'kind' column is lost. For example, even if the 'kind' values are in int32, in the resulting settings dict, the
        type of the top level keys (representing different kind values) will be str
        """
        df = pd.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [1, 1, 2, 2],
                "kind": [1, 2, 1, 2],
                "value": [1, 2, 3, 4],
            }
        )

        features = extract_features(
            df,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            column_value="value",
            default_fc_parameters=MinimalFCParameters(),
        )
        sample_settings = from_columns(features)
        X = extract_features(
            df,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            column_value="value",
            kind_to_fc_parameters=sample_settings,
        )
        assert X.shape == (1, 2 * len(MinimalFCParameters()))


class TestEfficientFCParameters(TestCase):
    """
    This tests the EfficientFCParameters( class
    """

    def test_extraction_runs_through(self):
        rfs = EfficientFCParameters()
        data = pd.DataFrame(
            [[0, 0, 0, 0], [1, 0, 0, 0]], columns=["id", "time", "kind", "value"]
        )

        extracted_features = extract_features(
            data,
            default_fc_parameters=rfs,
            column_kind="kind",
            column_value="value",
            column_sort="time",
            column_id="id",
        )

        self.assertCountEqual(extracted_features.index, [0, 1])

    def test_contains_all_non_high_comp_cost_features(self):
        """
        Test that by default a EfficientFCParameters object should be set up to calculate all features defined
        in tsfresh.feature_extraction.feature_calculators that do not have the attribute "high_comp_cost"
        """
        rfs = EfficientFCParameters()
        all_feature_calculators = [
            name
            for name, func in feature_calculators.__dict__.items()
            if hasattr(func, "fctype") and not hasattr(func, "high_comp_cost")
        ]

        for calculator in all_feature_calculators:
            self.assertIn(
                calculator,
                rfs,
                msg="Default EfficientFCParameters object does not setup calculation of {}".format(
                    calculator
                ),
            )

    def test_contains_all_time_based_features(self):
        """
        Test that by default a TimeBasedFCParameters object should be set up to calculate all
        features defined in tsfresh.feature_extraction.feature_calculators that have the
        attribute "index_type" == pd.DatetimeIndex
        """
        rfs = TimeBasedFCParameters()
        all_feature_calculators = [
            name
            for name, func in feature_calculators.__dict__.items()
            if not getattr(func, "index_type", False) != pd.DatetimeIndex
        ]

        for calculator in all_feature_calculators:
            self.assertIn(
                calculator,
                rfs,
                msg="Default TimeBasedFCParameters object does not setup calculation of {}".format(
                    calculator
                ),
            )

    def test_contains_all_index_based_features(self):
        """
        Test that by default a IndexBasedFCParameters object should be set up to calculate all
        features defined in tsfresh.feature_extraction.feature_calculators that have the
        attribute "input" == "pd.Series"
        """
        rfs = IndexBasedFCParameters()
        all_feature_calculators = [
            name
            for name, func in feature_calculators.__dict__.items()
            if getattr(func, "input", None) == "pd.Series"
        ]

        for calculator in all_feature_calculators:
            self.assertIn(
                calculator,
                rfs,
                msg="Default IndexBasedFCParameters object does not setup calculation "
                "of {}".format(calculator),
            )


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

        data = pd.DataFrame(
            [[0, 0, 0, 0], [1, 0, 0, 0]], columns=["id", "time", "kind", "value"]
        )

        extracted_features = extract_features(
            data,
            default_fc_parameters=mfs,
            column_kind="kind",
            column_value="value",
            column_sort="time",
            column_id="id",
        )

        self.assertCountEqual(
            extracted_features.columns,
            [
                "0__median",
                "0__standard_deviation",
                "0__sum_values",
                "0__maximum",
                "0__variance",
                "0__minimum",
                "0__mean",
                "0__length",
                "0__root_mean_square",
                "0__absolute_maximum",
            ],
        )
        self.assertCountEqual(extracted_features.index, [0, 1])


class TestSettingPickability(TestCase):
    def test_settings_pickable(self):
        settings = PickableSettings()
        settings["test"] = 3
        settings[lambda x: x + 1] = None

        def f(x):
            return x - 2

        settings[f] = {"this": "is a test"}

        dumped_settings = pickle.dumps(settings)
        settings = pickle.loads(dumped_settings)

        self.assertIn("test", settings)
        self.assertEqual(len(settings), 3)

        for key in settings:
            self.assertTrue(
                not callable(key)
                or (key(3) == 4 and settings[key] is None)
                or (key(3) == 1 and settings[key] == {"this": "is a test"})
            )
