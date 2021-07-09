# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import math
import warnings
from random import shuffle
from unittest import TestCase

from matrixprofile.exceptions import NoSolutionPossible

from tsfresh.examples.driftbif_simulation import velocity
from tsfresh.feature_extraction.feature_calculators import *
from tsfresh.feature_extraction.feature_calculators import (
    _aggregate_on_chunks,
    _estimate_friedrich_coefficients,
    _get_length_sequences_where,
    _into_subchunks,
    _roll,
)


class FeatureCalculationTestCase(TestCase):
    def setUp(self):
        # There will be a lot of warnings in the feature calculators.
        # Just ignore all of them in these tests
        warnings.simplefilter("ignore")

    def tearDown(self):
        warnings.resetwarnings()

    def assertIsNaN(self, result):
        self.assertTrue(np.isnan(result), msg="{} is not np.NaN")

    def assertEqualOnAllArrayTypes(self, f, input_to_f, result, *args, **kwargs):
        expected_result = f(input_to_f, *args, **kwargs)
        self.assertEqual(
            expected_result,
            result,
            msg="Not equal for lists: {} != {}".format(expected_result, result),
        )
        expected_result = f(np.array(input_to_f), *args, **kwargs)
        self.assertEqual(
            expected_result,
            result,
            msg="Not equal for numpy.arrays: {} != {}".format(expected_result, result),
        )
        expected_result = f(pd.Series(input_to_f, dtype="float64"), *args, **kwargs)
        self.assertEqual(
            expected_result,
            result,
            msg="Not equal for pandas.Series: {} != {}".format(expected_result, result),
        )

    def assertTrueOnAllArrayTypes(self, f, input_to_f, *args, **kwargs):
        self.assertTrue(f(input_to_f, *args, **kwargs), msg="Not true for lists")
        self.assertTrue(
            f(np.array(input_to_f), *args, **kwargs), msg="Not true for numpy.arrays"
        )
        self.assertTrue(
            f(pd.Series(input_to_f), *args, **kwargs), msg="Not true for pandas.Series"
        )

    def assertAllTrueOnAllArrayTypes(self, f, input_to_f, *args, **kwargs):
        self.assertTrue(
            all(dict(f(input_to_f, *args, **kwargs)).values()), msg="Not true for lists"
        )
        self.assertTrue(
            all(dict(f(np.array(input_to_f), *args, **kwargs)).values()),
            msg="Not true for numpy.arrays",
        )
        self.assertTrue(
            all(dict(f(pd.Series(input_to_f), *args, **kwargs)).values()),
            msg="Not true for pandas.Series",
        )

    def assertFalseOnAllArrayTypes(self, f, input_to_f, *args, **kwargs):
        self.assertFalse(f(input_to_f, *args, **kwargs), msg="Not false for lists")
        self.assertFalse(
            f(np.array(input_to_f), *args, **kwargs), msg="Not false for numpy.arrays"
        )
        self.assertFalse(
            f(pd.Series(input_to_f), *args, **kwargs), msg="Not false for pandas.Series"
        )

    def assertAllFalseOnAllArrayTypes(self, f, input_to_f, *args, **kwargs):
        self.assertFalse(
            any(dict(f(input_to_f, *args, **kwargs)).values()),
            msg="Not false for lists",
        )
        self.assertFalse(
            any(dict(f(np.array(input_to_f), *args, **kwargs)).values()),
            msg="Not false for numpy.arrays",
        )
        self.assertFalse(
            any(dict(f(pd.Series(input_to_f), *args, **kwargs)).values()),
            msg="Not false for pandas.Series",
        )

    def assertAlmostEqualOnAllArrayTypes(self, f, input_to_f, result, *args, **kwargs):
        expected_result = f(input_to_f, *args, **kwargs)
        self.assertAlmostEqual(
            expected_result,
            result,
            msg="Not almost equal for lists: {} != {}".format(expected_result, result),
        )
        expected_result = f(np.array(input_to_f), *args, **kwargs)
        self.assertAlmostEqual(
            expected_result,
            result,
            msg="Not almost equal for numpy.arrays: {} != {}".format(
                expected_result, result
            ),
        )
        expected_result = f(pd.Series(input_to_f, dtype="float64"), *args, **kwargs)
        self.assertAlmostEqual(
            expected_result,
            result,
            msg="Not almost equal for pandas.Series: {} != {}".format(
                expected_result, result
            ),
        )

    def assertIsNanOnAllArrayTypes(self, f, input_to_f, *args, **kwargs):
        self.assertTrue(
            np.isnan(f(input_to_f, *args, **kwargs)), msg="Not NaN for lists"
        )
        self.assertTrue(
            np.isnan(f(np.array(input_to_f), *args, **kwargs)),
            msg="Not NaN for numpy.arrays",
        )
        self.assertTrue(
            np.isnan(f(pd.Series(input_to_f, dtype="float64"), *args, **kwargs)),
            msg="Not NaN for pandas.Series",
        )

    def assertEqualPandasSeriesWrapper(self, f, input_to_f, result, *args, **kwargs):
        self.assertEqual(
            f(pd.Series(input_to_f), *args, **kwargs),
            result,
            msg="Not equal for pandas.Series: {} != {}".format(
                f(pd.Series(input_to_f), *args, **kwargs), result
            ),
        )

    def test__roll(self):
        x = np.random.normal(size=30)
        for shift in [0, 1, 10, 11, 30, 31, 50, 51, 150, 151]:
            np.testing.assert_array_equal(_roll(x, shift), np.roll(x, shift))
            np.testing.assert_array_equal(_roll(x, -shift), np.roll(x, -shift))

    def test___get_length_sequences_where(self):
        self.assertEqualOnAllArrayTypes(
            _get_length_sequences_where,
            [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
            [1, 3, 1, 2],
        )
        self.assertEqualOnAllArrayTypes(
            _get_length_sequences_where,
            [0, True, 0, 0, True, True, True, 0, 0, True, 0, True, True],
            [1, 3, 1, 2],
        )
        self.assertEqualOnAllArrayTypes(
            _get_length_sequences_where,
            [0, True, 0, 0, 1, True, 1, 0, 0, True, 0, 1, True],
            [1, 3, 1, 2],
        )
        self.assertEqualOnAllArrayTypes(_get_length_sequences_where, [0] * 10, [0])
        self.assertEqualOnAllArrayTypes(_get_length_sequences_where, [], [0])

    def test__into_subchunks(self):
        np.testing.assert_array_equal(
            _into_subchunks(range(7), 3, 2), np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]])
        )
        np.testing.assert_array_equal(
            _into_subchunks(range(5), 3), np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        )

    def test_variance_larger_than_standard_deviation(self):
        self.assertFalseOnAllArrayTypes(
            variance_larger_than_standard_deviation, [-1, -1, 1, 1, 1]
        )
        self.assertTrueOnAllArrayTypes(
            variance_larger_than_standard_deviation, [-1, -1, 1, 1, 2]
        )

    def test_large_standard_deviation(self):
        self.assertFalseOnAllArrayTypes(large_standard_deviation, [1, 1, 1, 1], r=0)
        self.assertFalseOnAllArrayTypes(large_standard_deviation, [1, 1, 1, 1], r=0)
        self.assertTrueOnAllArrayTypes(large_standard_deviation, [-1, -1, 1, 1], r=0)
        self.assertTrueOnAllArrayTypes(large_standard_deviation, [-1, -1, 1, 1], r=0.25)
        self.assertTrueOnAllArrayTypes(large_standard_deviation, [-1, -1, 1, 1], r=0.3)
        self.assertFalseOnAllArrayTypes(large_standard_deviation, [-1, -1, 1, 1], r=0.5)

    def test_symmetry_looking(self):
        self.assertAllTrueOnAllArrayTypes(
            symmetry_looking, [-1, -1, 1, 1], [dict(r=0.05), dict(r=0.75)]
        )
        self.assertAllFalseOnAllArrayTypes(
            symmetry_looking, [-1, -1, 1, 1], [dict(r=0)]
        )
        self.assertAllFalseOnAllArrayTypes(
            symmetry_looking, [-1, -1, -1, -1, 1], [dict(r=0.05)]
        )
        self.assertAllTrueOnAllArrayTypes(
            symmetry_looking, [-2, -2, -2, -1, -1, -1], [dict(r=0.05)]
        )
        self.assertAllTrueOnAllArrayTypes(
            symmetry_looking, [-0.9, -0.900001], [dict(r=0.05)]
        )

    def test_has_duplicate_max(self):
        self.assertTrueOnAllArrayTypes(has_duplicate_max, [2.1, 0, 0, 2.1, 1.1])
        self.assertFalseOnAllArrayTypes(
            has_duplicate_max, np.array([2.1, 0, 0, 2, 1.1])
        )
        self.assertTrueOnAllArrayTypes(has_duplicate_max, [1, 1, 1, 1])
        self.assertFalseOnAllArrayTypes(has_duplicate_max, np.array([0]))
        self.assertTrueOnAllArrayTypes(has_duplicate_max, np.array([1, 1]))

    def test_has_duplicate_min(self):
        self.assertTrueOnAllArrayTypes(has_duplicate_min, [-2.1, 0, 0, -2.1, 1.1])
        self.assertFalseOnAllArrayTypes(has_duplicate_min, [2.1, 0, -1, 2, 1.1])
        self.assertTrueOnAllArrayTypes(has_duplicate_min, np.array([1, 1, 1, 1]))
        self.assertFalseOnAllArrayTypes(has_duplicate_min, np.array([0]))
        self.assertTrueOnAllArrayTypes(has_duplicate_min, np.array([1, 1]))

    def test_has_duplicate(self):
        self.assertTrueOnAllArrayTypes(has_duplicate, np.array([-2.1, 0, 0, -2.1]))
        self.assertTrueOnAllArrayTypes(has_duplicate, [-2.1, 2.1, 2.1, 2.1])
        self.assertFalseOnAllArrayTypes(has_duplicate, [1.1, 1.2, 1.3, 1.4])
        self.assertFalseOnAllArrayTypes(has_duplicate, [1])
        self.assertFalseOnAllArrayTypes(has_duplicate, [])

    def test_sum(self):
        self.assertEqualOnAllArrayTypes(sum_values, [1, 2, 3, 4.1], 10.1)
        self.assertEqualOnAllArrayTypes(sum_values, [-1.2, -2, -3, -4], -10.2)
        self.assertEqualOnAllArrayTypes(sum_values, [], 0)

    def test_agg_autocorrelation_returns_correct_values(self):

        param = [{"f_agg": "mean", "maxlag": 10}]
        x = [1, 1, 1, 1, 1, 1, 1]
        expected_res = 0
        res = dict(agg_autocorrelation(x, param=param))['f_agg_"mean"__maxlag_10']
        self.assertAlmostEqual(res, expected_res, places=4)

        x = [1, 2, -3]
        expected_res = 1 / np.var(x) * (((1 * 2 + 2 * (-3)) / 2 + (1 * -3)) / 2)
        res = dict(agg_autocorrelation(x, param=param))['f_agg_"mean"__maxlag_10']
        self.assertAlmostEqual(res, expected_res, places=4)

        np.random.seed(42)
        x = np.random.normal(size=3000)
        expected_res = 0
        res = dict(agg_autocorrelation(x, param=param))['f_agg_"mean"__maxlag_10']
        self.assertAlmostEqual(res, expected_res, places=2)

        param = [{"f_agg": "median", "maxlag": 10}]
        x = [1, 1, 1, 1, 1, 1, 1]
        expected_res = 0
        res = dict(agg_autocorrelation(x, param=param))['f_agg_"median"__maxlag_10']
        self.assertAlmostEqual(res, expected_res, places=4)

        x = [1, 2, -3]
        expected_res = 1 / np.var(x) * (((1 * 2 + 2 * (-3)) / 2 + (1 * -3)) / 2)
        res = dict(agg_autocorrelation(x, param=param))['f_agg_"median"__maxlag_10']
        self.assertAlmostEqual(res, expected_res, places=4)

    def test_agg_autocorrelation_returns_max_lag_does_not_affect_other_results(self):

        param = [{"f_agg": "mean", "maxlag": 1}, {"f_agg": "mean", "maxlag": 10}]
        x = range(10)
        res1 = dict(agg_autocorrelation(x, param=param))['f_agg_"mean"__maxlag_1']
        res10 = dict(agg_autocorrelation(x, param=param))['f_agg_"mean"__maxlag_10']
        self.assertAlmostEqual(res1, 0.77777777, places=4)
        self.assertAlmostEqual(res10, -0.64983164983165, places=4)

        param = [{"f_agg": "mean", "maxlag": 1}]
        x = range(10)
        res1 = dict(agg_autocorrelation(x, param=param))['f_agg_"mean"__maxlag_1']
        self.assertAlmostEqual(res1, 0.77777777, places=4)

    def test_partial_autocorrelation(self):

        # Test for altering time series
        # len(x) < max_lag
        param = [{"lag": lag} for lag in range(10)]
        x = [1, 2, 1, 2, 1, 2]
        expected_res = [("lag_0", 1.0), ("lag_1", -1.0), ("lag_2", np.nan)]
        res = partial_autocorrelation(x, param=param)
        self.assertAlmostEqual(res[0][1], expected_res[0][1], places=4)
        self.assertAlmostEqual(res[1][1], expected_res[1][1], places=4)
        self.assertIsNaN(res[2][1])

        # Linear signal
        param = [{"lag": lag} for lag in range(10)]
        x = np.linspace(0, 1, 3000)
        expected_res = [("lag_0", 1.0), ("lag_1", 1.0), ("lag_2", 0)]
        res = partial_autocorrelation(x, param=param)
        self.assertAlmostEqual(res[0][1], expected_res[0][1], places=2)
        self.assertAlmostEqual(res[1][1], expected_res[1][1], places=2)
        self.assertAlmostEqual(res[2][1], expected_res[2][1], places=2)

        # Random noise
        np.random.seed(42)
        x = np.random.normal(size=3000)
        param = [{"lag": lag} for lag in range(10)]
        expected_res = [("lag_0", 1.0), ("lag_1", 0), ("lag_2", 0)]
        res = partial_autocorrelation(x, param=param)
        self.assertAlmostEqual(res[0][1], expected_res[0][1], places=1)
        self.assertAlmostEqual(res[1][1], expected_res[1][1], places=1)
        self.assertAlmostEqual(res[2][1], expected_res[2][1], places=1)

        # On a simulated AR process
        np.random.seed(42)
        param = [{"lag": lag} for lag in range(10)]
        # Simulate AR process
        T = 3000
        epsilon = np.random.randn(T)
        x = np.repeat(1.0, T)
        for t in range(T - 1):
            x[t + 1] = 0.5 * x[t] + 2 + epsilon[t]
        expected_res = [("lag_0", 1.0), ("lag_1", 0.5), ("lag_2", 0)]
        res = partial_autocorrelation(x, param=param)
        self.assertAlmostEqual(res[0][1], expected_res[0][1], places=1)
        self.assertAlmostEqual(res[1][1], expected_res[1][1], places=1)
        self.assertAlmostEqual(res[2][1], expected_res[2][1], places=1)

        # Some pathological cases
        param = [{"lag": lag} for lag in range(10)]
        # List of length 1
        res = partial_autocorrelation([1], param=param)
        for lag_no, lag_val in res:
            self.assertIsNaN(lag_val)
        # Empty list
        res = partial_autocorrelation([], param=param)
        for lag_no, lag_val in res:
            self.assertIsNaN(lag_val)
        # List contains only zeros
        res = partial_autocorrelation(np.zeros(100), param=param)
        for lag_no, lag_val in res:
            if lag_no == "lag_0":
                self.assertEqual(lag_val, 1.0)
            else:
                self.assertIsNaN(lag_val)

    def test_augmented_dickey_fuller(self):
        # todo: add unit test for the values of the test statistic

        # the adf hypothesis test checks for unit roots,
        # so H_0 = {random drift} vs H_1 = {AR(1) model}

        # H0 is true
        np.random.seed(seed=42)
        x = np.cumsum(np.random.uniform(size=100))
        param = [
            {"autolag": "BIC", "attr": "teststat"},
            {"autolag": "BIC", "attr": "pvalue"},
            {"autolag": "BIC", "attr": "usedlag"},
        ]
        expected_index = [
            'attr_"teststat"__autolag_"BIC"',
            'attr_"pvalue"__autolag_"BIC"',
            'attr_"usedlag"__autolag_"BIC"',
        ]

        res = augmented_dickey_fuller(x=x, param=param)
        res = pd.Series(dict(res))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertGreater(res['attr_"pvalue"__autolag_"BIC"'], 0.10)
        self.assertEqual(res['attr_"usedlag"__autolag_"BIC"'], 0)

        # H0 should be rejected for AR(1) model with x_{t} = 1/2 x_{t-1} + e_{t}
        np.random.seed(seed=42)
        e = np.random.normal(0.1, 0.1, size=100)
        m = 50
        x = [0] * m
        x[0] = 100
        for i in range(1, m):
            x[i] = x[i - 1] * 0.5 + e[i]
        param = [
            {"autolag": "AIC", "attr": "teststat"},
            {"autolag": "AIC", "attr": "pvalue"},
            {"autolag": "AIC", "attr": "usedlag"},
        ]
        expected_index = [
            'attr_"teststat"__autolag_"AIC"',
            'attr_"pvalue"__autolag_"AIC"',
            'attr_"usedlag"__autolag_"AIC"',
        ]

        res = augmented_dickey_fuller(x=x, param=param)
        res = pd.Series(dict(res))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertLessEqual(res['attr_"pvalue"__autolag_"AIC"'], 0.05)
        self.assertEqual(res['attr_"usedlag"__autolag_"AIC"'], 0)

        # Check if LinAlgError and ValueError are catched
        res_linalg_error = augmented_dickey_fuller(
            x=np.repeat(np.nan, 100), param=param
        )
        res_value_error = augmented_dickey_fuller(x=[], param=param)
        for index, val in res_linalg_error:
            self.assertIsNaN(val)
        for index, val in res_value_error:
            self.assertIsNaN(val)

        # Should return NaN if "attr" is unknown
        res_attr_error = augmented_dickey_fuller(
            x=x, param=[{"autolag": "AIC", "attr": ""}]
        )
        for index, val in res_attr_error:
            self.assertIsNaN(val)

    def test_abs_energy(self):
        self.assertEqualOnAllArrayTypes(abs_energy, [1, 1, 1], 3)
        self.assertEqualOnAllArrayTypes(abs_energy, [1, 2, 3], 14)
        self.assertEqualOnAllArrayTypes(abs_energy, [-1, 2, -3], 14)
        self.assertAlmostEqualOnAllArrayTypes(abs_energy, [-1, 1.3], 2.69)
        self.assertEqualOnAllArrayTypes(abs_energy, [1], 1)

    def test_cid_ce(self):
        self.assertEqualOnAllArrayTypes(cid_ce, [1, 1, 1], 0, normalize=True)
        self.assertEqualOnAllArrayTypes(cid_ce, [0, 4], 2, normalize=True)
        self.assertEqualOnAllArrayTypes(cid_ce, [100, 104], 2, normalize=True)

        self.assertEqualOnAllArrayTypes(cid_ce, [1, 1, 1], 0, normalize=False)
        self.assertEqualOnAllArrayTypes(cid_ce, [0.5, 3.5, 7.5], 5, normalize=False)
        self.assertEqualOnAllArrayTypes(
            cid_ce, [-4.33, -1.33, 2.67], 5, normalize=False
        )

    def test_lempel_ziv_complexity(self):
        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, [1, 1, 1], 2.0 / 3, bins=2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, [1, 1, 1], 2.0 / 3, bins=5
        )

        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, [1, 1, 1, 1, 1, 1, 1], 0.4285714285, bins=2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, [1, 1, 1, 2, 1, 1, 1], 0.5714285714, bins=2
        )

        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, [-1, 4.3, 5, 1, -4.5, 1, 5, 7, -3.4, 6], 0.8, bins=10
        )
        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity,
            [-1, np.nan, 5, 1, -4.5, 1, 5, 7, -3.4, 6],
            0.4,
            bins=10,
        )
        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, np.linspace(0, 1, 10), 0.6, bins=3
        )
        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, [1, 1, 2, 3, 4, 5, 6, 0, 7, 8], 0.6, bins=3
        )

    def test_fourier_entropy(self):
        self.assertAlmostEqualOnAllArrayTypes(
            fourier_entropy, [1, 2, 1], 0.693147180, bins=2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            fourier_entropy, [1, 2, 1], 0.693147180, bins=5
        )

        self.assertAlmostEqualOnAllArrayTypes(
            fourier_entropy, [1, 1, 2, 1, 1, 1, 1], 0.5623351446188083, bins=5
        )
        self.assertAlmostEqualOnAllArrayTypes(
            fourier_entropy, [1, 1, 1, 1, 2, 1, 1], 1.0397207708399179, bins=5
        )

        self.assertAlmostEqualOnAllArrayTypes(
            fourier_entropy,
            [-1, 4.3, 5, 1, -4.5, 1, 5, 7, -3.4, 6],
            1.5607104090414063,
            bins=10,
        )
        self.assertIsNanOnAllArrayTypes(
            fourier_entropy, [-1, np.nan, 5, 1, -4.5, 1, 5, 7, -3.4, 6], bins=10
        )

    def test_permutation_entropy(self):
        self.assertAlmostEqualOnAllArrayTypes(
            permutation_entropy,
            [4, 7, 9, 10, 6, 11, 3],
            1.054920167,
            dimension=3,
            tau=1,
        )

        # should grow
        self.assertAlmostEqualOnAllArrayTypes(
            permutation_entropy,
            [1, -1, 1, -1, 1, -1, 1, -1],
            0.6931471805599453,
            dimension=3,
            tau=1,
        )
        self.assertAlmostEqualOnAllArrayTypes(
            permutation_entropy,
            [1, -1, 1, -1, 1, 1, 1, -1],
            1.3296613488547582,
            dimension=3,
            tau=1,
        )

        self.assertAlmostEqualOnAllArrayTypes(
            permutation_entropy,
            [-1, 4.3, 5, 1, -4.5, 1, 5, 7, -3.4, 6],
            1.0397207708399179,
            dimension=3,
            tau=2,
        )
        # nan is treated like any other number
        self.assertAlmostEqualOnAllArrayTypes(
            permutation_entropy,
            [-1, 4.3, 5, 1, -4.5, 1, 5, np.nan, -3.4, 6],
            1.0397207708399179,
            dimension=3,
            tau=2,
        )

        # if too short, return nan
        self.assertIsNanOnAllArrayTypes(
            permutation_entropy, [1, -1], dimension=3, tau=1
        )

    def test_ratio_beyond_r_sigma(self):

        x = [0, 1] * 10 + [10, 20, -30]  # std of x is 7.21, mean 3.04
        self.assertEqualOnAllArrayTypes(ratio_beyond_r_sigma, x, 3.0 / len(x), r=1)
        self.assertEqualOnAllArrayTypes(ratio_beyond_r_sigma, x, 2.0 / len(x), r=2)
        self.assertEqualOnAllArrayTypes(ratio_beyond_r_sigma, x, 1.0 / len(x), r=3)
        self.assertEqualOnAllArrayTypes(ratio_beyond_r_sigma, x, 0, r=20)

    def test_mean_abs_change(self):
        self.assertEqualOnAllArrayTypes(mean_abs_change, [-2, 2, 5], 3.5)
        self.assertEqualOnAllArrayTypes(mean_abs_change, [1, 2, -1], 2)

    def test_mean_change(self):
        self.assertEqualOnAllArrayTypes(mean_change, [-2, 2, 5], 3.5)
        self.assertEqualOnAllArrayTypes(mean_change, [1, 2, -1], -1)
        self.assertEqualOnAllArrayTypes(mean_change, [10, 20], 10)
        self.assertIsNanOnAllArrayTypes(mean_change, [1])
        self.assertIsNanOnAllArrayTypes(mean_change, [])

    def test_mean_second_derivate_central(self):
        self.assertEqualOnAllArrayTypes(
            mean_second_derivative_central, list(range(10)), 0
        )
        self.assertEqualOnAllArrayTypes(mean_second_derivative_central, [1, 3, 5], 0)
        self.assertEqualOnAllArrayTypes(
            mean_second_derivative_central, [1, 3, 7, -3], -3
        )

    def test_median(self):
        self.assertEqualOnAllArrayTypes(median, [1, 1, 2, 2], 1.5)
        self.assertEqualOnAllArrayTypes(median, [0.5, 0.5, 2, 3.5, 10], 2)
        self.assertEqualOnAllArrayTypes(median, [0.5], 0.5)
        self.assertIsNanOnAllArrayTypes(median, [])

    def test_mean(self):
        self.assertEqualOnAllArrayTypes(mean, [1, 1, 2, 2], 1.5)
        self.assertEqualOnAllArrayTypes(mean, [0.5, 0.5, 2, 3.5, 10], 3.3)
        self.assertEqualOnAllArrayTypes(mean, [0.5], 0.5)
        self.assertIsNanOnAllArrayTypes(mean, [])

    def test_length(self):
        self.assertEqualOnAllArrayTypes(length, [1, 2, 3, 4], 4)
        self.assertEqualOnAllArrayTypes(length, [1, 2, 3], 3)
        self.assertEqualOnAllArrayTypes(length, [1, 2], 2)
        self.assertEqualOnAllArrayTypes(length, [1, 2, 3, np.NaN], 4)
        self.assertEqualOnAllArrayTypes(length, [], 0)

    def test_standard_deviation(self):
        self.assertAlmostEqualOnAllArrayTypes(standard_deviation, [1, 1, -1, -1], 1)
        self.assertAlmostEqualOnAllArrayTypes(
            standard_deviation, [1, 2, -2, -1], 1.58113883008
        )
        self.assertIsNanOnAllArrayTypes(standard_deviation, [])

    def test_variation_coefficient(self):
        self.assertIsNanOnAllArrayTypes(
            variation_coefficient, [1, 1, -1, -1],
        )
        self.assertAlmostEqualOnAllArrayTypes(
            variation_coefficient, [1, 2, -3, -1], -7.681145747868608
        )
        self.assertAlmostEqualOnAllArrayTypes(
            variation_coefficient, [1, 2, 4, -1], 1.2018504251546631
        )
        self.assertIsNanOnAllArrayTypes(variation_coefficient, [])

    def test_variance(self):
        self.assertAlmostEqualOnAllArrayTypes(variance, [1, 1, -1, -1], 1)
        self.assertAlmostEqualOnAllArrayTypes(variance, [1, 2, -2, -1], 2.5)
        self.assertIsNanOnAllArrayTypes(variance, [])

    def test_skewness(self):
        self.assertEqualOnAllArrayTypes(skewness, [1, 1, 1, 2, 2, 2], 0)
        self.assertAlmostEqualOnAllArrayTypes(
            skewness, [1, 1, 1, 2, 2], 0.6085806194501855
        )
        self.assertEqualOnAllArrayTypes(skewness, [1, 1, 1], 0)
        self.assertIsNanOnAllArrayTypes(skewness, [1, 1])

    def test_kurtosis(self):
        self.assertAlmostEqualOnAllArrayTypes(
            kurtosis, [1, 1, 1, 2, 2], -3.333333333333333
        )
        self.assertAlmostEqualOnAllArrayTypes(kurtosis, [1, 1, 1, 1], 0)
        self.assertIsNanOnAllArrayTypes(kurtosis, [1, 1, 1])

    def test_root_mean_square(self):
        self.assertAlmostEqualOnAllArrayTypes(
            root_mean_square, [1, 1, 1, 2, 2], 1.4832396974191
        )
        self.assertAlmostEqualOnAllArrayTypes(root_mean_square, [0], 0)
        self.assertIsNanOnAllArrayTypes(root_mean_square, [])
        self.assertAlmostEqualOnAllArrayTypes(root_mean_square, [1], 1)
        self.assertAlmostEqualOnAllArrayTypes(root_mean_square, [-1], 1)

    def test_mean_n_absolute_max(self):
        self.assertIsNanOnAllArrayTypes(mean_n_absolute_max, [], number_of_maxima=1)
        self.assertIsNanOnAllArrayTypes(
            mean_n_absolute_max, [12, 3], number_of_maxima=10
        )
        self.assertRaises(
            AssertionError, mean_n_absolute_max, [12, 3], number_of_maxima=0
        )
        self.assertRaises(
            AssertionError, mean_n_absolute_max, [12, 3], number_of_maxima=-1
        )
        self.assertAlmostEqualOnAllArrayTypes(
            mean_n_absolute_max, [-1, -5, 4, 10], 6.33333333333, number_of_maxima=3
        )
        self.assertAlmostEqualOnAllArrayTypes(
            mean_n_absolute_max, [0, -5, -9], 7.000000, number_of_maxima=2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            mean_n_absolute_max, [0, 0, 0], 0, number_of_maxima=1
        )

    def test_absolute_sum_of_changes(self):
        self.assertEqualOnAllArrayTypes(absolute_sum_of_changes, [1, 1, 1, 1, 2, 1], 2)
        self.assertEqualOnAllArrayTypes(absolute_sum_of_changes, [1, -1, 1, -1], 6)
        self.assertEqualOnAllArrayTypes(absolute_sum_of_changes, [1], 0)
        self.assertEqualOnAllArrayTypes(absolute_sum_of_changes, [], 0)

    def test_longest_strike_below_mean(self):
        self.assertEqualOnAllArrayTypes(
            longest_strike_below_mean, [1, 2, 1, 1, 1, 2, 2, 2], 3
        )
        self.assertEqualOnAllArrayTypes(
            longest_strike_below_mean, [1, 2, 3, 4, 5, 6], 3
        )
        self.assertEqualOnAllArrayTypes(longest_strike_below_mean, [1, 2, 3, 4, 5], 2)
        self.assertEqualOnAllArrayTypes(longest_strike_below_mean, [1, 2, 1], 1)
        self.assertEqualOnAllArrayTypes(longest_strike_below_mean, [], 0)

    def test_longest_strike_above_mean(self):
        self.assertEqualOnAllArrayTypes(
            longest_strike_above_mean, [1, 2, 1, 2, 1, 2, 2, 1], 2
        )
        self.assertEqualOnAllArrayTypes(
            longest_strike_above_mean, [1, 2, 3, 4, 5, 6], 3
        )
        self.assertEqualOnAllArrayTypes(longest_strike_above_mean, [1, 2, 3, 4, 5], 2)
        self.assertEqualOnAllArrayTypes(longest_strike_above_mean, [1, 2, 1], 1)
        self.assertEqualOnAllArrayTypes(longest_strike_above_mean, [], 0)

    def test_count_above_mean(self):
        self.assertEqualOnAllArrayTypes(count_above_mean, [1, 2, 1, 2, 1, 2], 3)
        self.assertEqualOnAllArrayTypes(count_above_mean, [1, 1, 1, 1, 1, 2], 1)
        self.assertEqualOnAllArrayTypes(count_above_mean, [1, 1, 1, 1, 1], 0)
        self.assertEqualOnAllArrayTypes(count_above_mean, [], 0)

    def test_count_below_mean(self):
        self.assertEqualOnAllArrayTypes(count_below_mean, [1, 2, 1, 2, 1, 2], 3)
        self.assertEqualOnAllArrayTypes(count_below_mean, [1, 1, 1, 1, 1, 2], 5)
        self.assertEqualOnAllArrayTypes(count_below_mean, [1, 1, 1, 1, 1], 0)
        self.assertEqualOnAllArrayTypes(count_below_mean, [], 0)

    def test_last_location_maximum(self):
        self.assertAlmostEqualOnAllArrayTypes(
            last_location_of_maximum, [1, 2, 1, 2, 1], 0.8
        )
        self.assertAlmostEqualOnAllArrayTypes(
            last_location_of_maximum, [1, 2, 1, 1, 2], 1.0
        )
        self.assertAlmostEqualOnAllArrayTypes(
            last_location_of_maximum, [2, 1, 1, 1, 1], 0.2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            last_location_of_maximum, [1, 1, 1, 1, 1], 1.0
        )
        self.assertAlmostEqualOnAllArrayTypes(last_location_of_maximum, [1], 1.0)
        self.assertIsNanOnAllArrayTypes(last_location_of_maximum, [])

    def test_first_location_of_maximum(self):
        self.assertAlmostEqualOnAllArrayTypes(
            first_location_of_maximum, [1, 2, 1, 2, 1], 0.2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            first_location_of_maximum, [1, 2, 1, 1, 2], 0.2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            first_location_of_maximum, [2, 1, 1, 1, 1], 0.0
        )
        self.assertAlmostEqualOnAllArrayTypes(
            first_location_of_maximum, [1, 1, 1, 1, 1], 0.0
        )
        self.assertAlmostEqualOnAllArrayTypes(first_location_of_maximum, [1], 0.0)
        self.assertIsNanOnAllArrayTypes(first_location_of_maximum, [])

    def test_last_location_of_minimum(self):
        self.assertAlmostEqualOnAllArrayTypes(
            last_location_of_minimum, [1, 2, 1, 2, 1], 1.0
        )
        self.assertAlmostEqualOnAllArrayTypes(
            last_location_of_minimum, [1, 2, 1, 2, 2], 0.6
        )
        self.assertAlmostEqualOnAllArrayTypes(
            last_location_of_minimum, [2, 1, 1, 1, 2], 0.8
        )
        self.assertAlmostEqualOnAllArrayTypes(
            last_location_of_minimum, [1, 1, 1, 1, 1], 1.0
        )
        self.assertAlmostEqualOnAllArrayTypes(last_location_of_minimum, [1], 1.0)
        self.assertIsNanOnAllArrayTypes(last_location_of_minimum, [])

    def test_first_location_of_minimum(self):
        self.assertAlmostEqualOnAllArrayTypes(
            first_location_of_minimum, [1, 2, 1, 2, 1], 0.0
        )
        self.assertAlmostEqualOnAllArrayTypes(
            first_location_of_minimum, [2, 2, 1, 2, 2], 0.4
        )
        self.assertAlmostEqualOnAllArrayTypes(
            first_location_of_minimum, [2, 1, 1, 1, 2], 0.2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            first_location_of_minimum, [1, 1, 1, 1, 1], 0.0
        )
        self.assertAlmostEqualOnAllArrayTypes(first_location_of_minimum, [1], 0.0)
        self.assertIsNanOnAllArrayTypes(first_location_of_minimum, [])

    def test_percentage_of_doubled_datapoints(self):
        self.assertAlmostEqualOnAllArrayTypes(
            percentage_of_reoccurring_datapoints_to_all_datapoints, [1, 1, 2, 3, 4], 0.4
        )
        self.assertAlmostEqualOnAllArrayTypes(
            percentage_of_reoccurring_datapoints_to_all_datapoints, [1, 1.5, 2, 3], 0
        )
        self.assertAlmostEqualOnAllArrayTypes(
            percentage_of_reoccurring_datapoints_to_all_datapoints, [1], 0
        )
        self.assertAlmostEqualOnAllArrayTypes(
            percentage_of_reoccurring_datapoints_to_all_datapoints,
            [1.111, -2.45, 1.111, 2.45],
            0.5,
        )
        self.assertIsNanOnAllArrayTypes(
            percentage_of_reoccurring_datapoints_to_all_datapoints, []
        )

    def test_ratio_of_doubled_values(self):
        self.assertAlmostEqualOnAllArrayTypes(
            percentage_of_reoccurring_values_to_all_values, [1, 1, 2, 3, 4], 0.25
        )
        self.assertAlmostEqualOnAllArrayTypes(
            percentage_of_reoccurring_values_to_all_values, [1, 1.5, 2, 3], 0
        )
        self.assertAlmostEqualOnAllArrayTypes(
            percentage_of_reoccurring_values_to_all_values, [1], 0
        )
        self.assertAlmostEqualOnAllArrayTypes(
            percentage_of_reoccurring_values_to_all_values,
            [1.111, -2.45, 1.111, 2.45],
            1.0 / 3.0,
        )
        self.assertIsNanOnAllArrayTypes(
            percentage_of_reoccurring_values_to_all_values, []
        )

    def test_sum_of_reoccurring_values(self):
        self.assertAlmostEqualOnAllArrayTypes(
            sum_of_reoccurring_values, [1, 1, 2, 3, 4, 4], 5
        )
        self.assertAlmostEqualOnAllArrayTypes(
            sum_of_reoccurring_values, [1, 1.5, 2, 3], 0
        )
        self.assertAlmostEqualOnAllArrayTypes(sum_of_reoccurring_values, [1], 0)
        self.assertAlmostEqualOnAllArrayTypes(
            sum_of_reoccurring_values, [1.111, -2.45, 1.111, 2.45], 1.111
        )
        self.assertAlmostEqualOnAllArrayTypes(sum_of_reoccurring_values, [], 0)

    def test_sum_of_reoccurring_data_points(self):
        self.assertAlmostEqualOnAllArrayTypes(
            sum_of_reoccurring_data_points, [1, 1, 2, 3, 4, 4], 10
        )
        self.assertAlmostEqualOnAllArrayTypes(
            sum_of_reoccurring_data_points, [1, 1.5, 2, 3], 0
        )
        self.assertAlmostEqualOnAllArrayTypes(sum_of_reoccurring_data_points, [1], 0)
        self.assertAlmostEqualOnAllArrayTypes(
            sum_of_reoccurring_data_points, [1.111, -2.45, 1.111, 2.45], 2.222
        )
        self.assertAlmostEqualOnAllArrayTypes(sum_of_reoccurring_data_points, [], 0)

    def test_uniqueness_factor(self):
        self.assertAlmostEqualOnAllArrayTypes(
            ratio_value_number_to_time_series_length, [1, 1, 2, 3, 4], 0.8
        )
        self.assertAlmostEqualOnAllArrayTypes(
            ratio_value_number_to_time_series_length, [1, 1.5, 2, 3], 1
        )
        self.assertAlmostEqualOnAllArrayTypes(
            ratio_value_number_to_time_series_length, [1], 1
        )
        self.assertAlmostEqualOnAllArrayTypes(
            ratio_value_number_to_time_series_length, [1.111, -2.45, 1.111, 2.45], 0.75
        )
        self.assertIsNanOnAllArrayTypes(ratio_value_number_to_time_series_length, [])

    def test_fft_coefficient(self):
        x = range(10)
        param = [
            {"coeff": 0, "attr": "real"},
            {"coeff": 1, "attr": "real"},
            {"coeff": 2, "attr": "real"},
            {"coeff": 0, "attr": "imag"},
            {"coeff": 1, "attr": "imag"},
            {"coeff": 2, "attr": "imag"},
            {"coeff": 0, "attr": "angle"},
            {"coeff": 1, "attr": "angle"},
            {"coeff": 2, "attr": "angle"},
            {"coeff": 0, "attr": "abs"},
            {"coeff": 1, "attr": "abs"},
            {"coeff": 2, "attr": "abs"},
        ]
        expected_index = [
            'attr_"real"__coeff_0',
            'attr_"real"__coeff_1',
            'attr_"real"__coeff_2',
            'attr_"imag"__coeff_0',
            'attr_"imag"__coeff_1',
            'attr_"imag"__coeff_2',
            'attr_"angle"__coeff_0',
            'attr_"angle"__coeff_1',
            'attr_"angle"__coeff_2',
            'attr_"abs"__coeff_0',
            'attr_"abs"__coeff_1',
            'attr_"abs"__coeff_2',
        ]

        res = pd.Series(dict(fft_coefficient(x, param)))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res['attr_"imag"__coeff_0'], 0, places=6)
        self.assertAlmostEqual(res['attr_"real"__coeff_0'], sum(x), places=6)
        self.assertAlmostEqual(res['attr_"angle"__coeff_0'], 0, places=6)
        self.assertAlmostEqual(res['attr_"abs"__coeff_0'], sum(x), places=6)

        x = [0, 1, 0, 0]
        res = pd.Series(dict(fft_coefficient(x, param)))
        # see documentation of fft in numpy
        # should return array([1. + 0.j, 0. - 1.j, -1. + 0.j])
        self.assertAlmostEqual(res['attr_"imag"__coeff_0'], 0, places=6)
        self.assertAlmostEqual(res['attr_"real"__coeff_0'], 1, places=6)
        self.assertAlmostEqual(res['attr_"imag"__coeff_1'], -1, places=6)
        self.assertAlmostEqual(res['attr_"angle"__coeff_1'], -90, places=6)
        self.assertAlmostEqual(res['attr_"real"__coeff_1'], 0, places=6)
        self.assertAlmostEqual(res['attr_"imag"__coeff_2'], 0, places=6)
        self.assertAlmostEqual(res['attr_"real"__coeff_2'], -1, places=6)

        # test what happens if coeff is biger than time series lenght
        x = range(5)
        param = [{"coeff": 10, "attr": "real"}]
        expected_index = ['attr_"real"__coeff_10']

        res = pd.Series(dict(fft_coefficient(x, param)))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertIsNaN(res['attr_"real"__coeff_10'])

    def test_fft_aggregated(self):
        param = [
            {"aggtype": "centroid"},
            {"aggtype": "variance"},
            {"aggtype": "skew"},
            {"aggtype": "kurtosis"},
        ]
        expected_index = [
            'aggtype_"centroid"',
            'aggtype_"variance"',
            'aggtype_"skew"',
            'aggtype_"kurtosis"',
        ]

        x = np.arange(10)
        res = pd.Series(dict(fft_aggregated(x, param)))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res['aggtype_"centroid"'], 1.135, places=3)
        self.assertAlmostEqual(res['aggtype_"variance"'], 2.368, places=3)
        self.assertAlmostEqual(res['aggtype_"skew"'], 1.249, places=3)
        self.assertAlmostEqual(res['aggtype_"kurtosis"'], 3.643, places=3)

        # Scalar multiplying the distribution should not change the results:
        x = 10 * x
        res = pd.Series(dict(fft_aggregated(x, param)))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res['aggtype_"centroid"'], 1.135, places=3)
        self.assertAlmostEqual(res['aggtype_"variance"'], 2.368, places=3)
        self.assertAlmostEqual(res['aggtype_"skew"'], 1.249, places=3)
        self.assertAlmostEqual(res['aggtype_"kurtosis"'], 3.643, places=3)

        # The fft of a sign wave is a dirac delta, variance and skew should be near zero, kurtosis should be near 3:
        # However, in the discrete limit, skew and kurtosis blow up in a manner that is noise dependent and are
        # therefore bad features, therefore an nan should be returned for these values
        x = np.sin(2 * np.pi / 10 * np.arange(30))
        res = pd.Series(dict(fft_aggregated(x, param)))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res['aggtype_"centroid"'], 3.0, places=5)
        self.assertAlmostEqual(res['aggtype_"variance"'], 0.0, places=5)
        self.assertIsNaN(res['aggtype_"skew"'])
        self.assertIsNaN(res['aggtype_"kurtosis"'])

        # Gaussian test:
        def normal(y, mean_, sigma_):
            return (
                1
                / (2 * np.pi * sigma_ ** 2)
                * np.exp(-((y - mean_) ** 2) / (2 * sigma_ ** 2))
            )

        mean_ = 500.0
        sigma_ = 1.0
        range_ = int(2 * mean_)
        x = list(map(lambda x: normal(x, mean_, sigma_), range(range_)))

        # The fourier transform of a Normal dist in the positive halfspace is a half normal,
        # Hand calculated values of centroid and variance based for the half-normal dist:
        # (Ref: https://en.wikipedia.org/wiki/Half-normal_distribution)
        expected_fft_centroid = (range_ / (2 * np.pi * sigma_)) * np.sqrt(2 / np.pi)
        expected_fft_var = (range_ / (2 * np.pi * sigma_)) ** 2 * (1 - 2 / np.pi)

        # Calculate values for unit test:
        res = pd.Series(dict(fft_aggregated(x, param)))
        self.assertCountEqual(list(res.index), expected_index)

        # Compare against hand calculated values:
        rel_diff_allowed = 0.02
        self.assertAlmostEqual(
            res['aggtype_"centroid"'],
            expected_fft_centroid,
            delta=rel_diff_allowed * expected_fft_centroid,
        )
        self.assertAlmostEqual(
            res['aggtype_"variance"'],
            expected_fft_var,
            delta=rel_diff_allowed * expected_fft_var,
        )

    def test_number_peaks(self):
        x = np.array([0, 1, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1])
        self.assertEqualOnAllArrayTypes(number_peaks, x, 2, 1)
        self.assertEqualOnAllArrayTypes(number_peaks, x, 2, 2)
        self.assertEqualOnAllArrayTypes(number_peaks, x, 1, 3)
        self.assertEqualOnAllArrayTypes(number_peaks, x, 1, 4)
        self.assertEqualOnAllArrayTypes(number_peaks, x, 0, 5)
        self.assertEqualOnAllArrayTypes(number_peaks, x, 0, 6)

    def test_mass_quantile(self):

        x = [1] * 101
        param = [{"q": 0.5}]
        expected_index = ["q_0.5"]
        res = index_mass_quantile(x, param)

        res = pd.Series(dict(res))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res["q_0.5"], 0.5, places=1)

        # Test for parts of pandas series
        x = pd.Series([0] * 55 + [1] * 101)
        param = [{"q": 0.5}]
        expected_index = ["q_0.5"]
        res = index_mass_quantile(x[x > 0], param)

        res = pd.Series(dict(res))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res["q_0.5"], 0.5, places=1)

        x = [0] * 1000 + [1]
        param = [{"q": 0.5}, {"q": 0.99}]
        expected_index = ["q_0.5", "q_0.99"]
        res = index_mass_quantile(x, param)

        res = pd.Series(dict(res))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res["q_0.5"], 1, places=1)
        self.assertAlmostEqual(res["q_0.99"], 1, places=1)

        x = [0, 1, 1, 0, 0, 1, 0, 0]
        param = [{"q": 0.30}, {"q": 0.60}, {"q": 0.90}]
        expected_index = ["q_0.3", "q_0.6", "q_0.9"]
        res = index_mass_quantile(x, param)

        res = pd.Series(dict(res))

        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res["q_0.3"], 0.25, places=1)
        self.assertAlmostEqual(res["q_0.6"], 0.375, places=1)
        self.assertAlmostEqual(res["q_0.9"], 0.75, places=1)

        x = [0, 0, 0]
        param = [{"q": 0.5}]
        expected_index = ["q_0.5"]
        res = index_mass_quantile(x, param)

        res = pd.Series(dict(res))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertTrue(np.isnan(res["q_0.5"]))

        x = []
        param = [{"q": 0.5}]
        expected_index = ["q_0.5"]
        res = index_mass_quantile(x, param)

        res = pd.Series(dict(res))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertTrue(np.isnan(res["q_0.5"]))

    def test_number_cwt_peaks(self):
        x = [1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1]
        self.assertEqualOnAllArrayTypes(number_cwt_peaks, x, 2, 2)

    def test_spkt_welch_density(self):

        # todo: improve tests
        x = range(10)
        param = [{"coeff": 1}, {"coeff": 10}]
        expected_index = ["coeff_1", "coeff_10"]
        res = pd.Series(dict(spkt_welch_density(x, param)))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertIsNaN(res["coeff_10"])

    def test_cwt_coefficients(self):
        x = [0.1, 0.2, 0.3]
        param = [
            {"widths": (1, 2, 3), "coeff": 2, "w": 1},
            {"widths": (1, 3), "coeff": 2, "w": 3},
            {"widths": (1, 3), "coeff": 5, "w": 3},
        ]
        shuffle(param)

        expected_index = [
            "coeff_2__w_1__widths_(1, 2, 3)",
            "coeff_2__w_3__widths_(1, 3)",
            "coeff_5__w_3__widths_(1, 3)",
        ]

        res = cwt_coefficients(x, param)
        res = pd.Series(dict(res))

        # todo: add unit test for the values
        self.assertCountEqual(list(res.index), expected_index)
        self.assertTrue(math.isnan(res["coeff_5__w_3__widths_(1, 3)"]))

    def test_ar_coefficient(self):

        # Test for X_i = 2.5 * X_{i-1} + 1
        param = [{"k": 1, "coeff": 0}, {"k": 1, "coeff": 1}]
        shuffle(param)

        x = [1] + 9 * [0]
        for i in range(1, len(x)):
            x[i] = 2.5 * x[i - 1] + 1

        res = ar_coefficient(x, param)
        expected_index = ["coeff_0__k_1", "coeff_1__k_1"]

        res = pd.Series(dict(res))
        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res["coeff_0__k_1"], 1, places=2)
        self.assertAlmostEqual(res["coeff_1__k_1"], 2.5, places=2)

        # Test for X_i = 1.4 * X_{i-1} - 1 X_{i-2} + 1
        param = [
            {"k": 1, "coeff": 0},
            {"k": 1, "coeff": 1},
            {"k": 2, "coeff": 0},
            {"k": 2, "coeff": 1},
            {"k": 2, "coeff": 2},
            {"k": 2, "coeff": 3},
        ]
        shuffle(param)

        x = [1, 1] + 5 * [0]
        for i in range(2, len(x)):
            x[i] = (-2) * x[i - 2] + 3.5 * x[i - 1] + 1

        res = ar_coefficient(x, param)
        expected_index = [
            "coeff_0__k_1",
            "coeff_1__k_1",
            "coeff_0__k_2",
            "coeff_1__k_2",
            "coeff_2__k_2",
            "coeff_3__k_2",
        ]

        res = pd.Series(dict(res))

        self.assertIsInstance(res, pd.Series)
        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res["coeff_0__k_2"], 1, places=2)
        self.assertAlmostEqual(res["coeff_1__k_2"], 3.5, places=2)
        self.assertAlmostEqual(res["coeff_2__k_2"], -2, places=2)
        self.assertTrue(np.isnan(res["coeff_3__k_2"]))

    def test_time_reversal_asymmetry_statistic(self):
        x = [1] * 10
        self.assertAlmostEqualOnAllArrayTypes(
            time_reversal_asymmetry_statistic, x, 0, 0
        )
        self.assertAlmostEqualOnAllArrayTypes(
            time_reversal_asymmetry_statistic, x, 0, 1
        )
        self.assertAlmostEqualOnAllArrayTypes(
            time_reversal_asymmetry_statistic, x, 0, 2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            time_reversal_asymmetry_statistic, x, 0, 3
        )

        x = [1, 2, -3, 4]
        # 1/2 * ( (4^2 * -3 + 3 * 2^2) + (3^2*2)-(2*1^1)) = 1/2 * (-48+12+18-2) = 20/2
        self.assertAlmostEqualOnAllArrayTypes(
            time_reversal_asymmetry_statistic, x, -10, 1
        )
        self.assertAlmostEqualOnAllArrayTypes(
            time_reversal_asymmetry_statistic, x, 0, 2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            time_reversal_asymmetry_statistic, x, 0, 3
        )

    def test_number_crossing_m(self):
        x = [10, -10, 10, -10]
        self.assertEqualOnAllArrayTypes(number_crossing_m, x, 3, 0)
        self.assertEqualOnAllArrayTypes(number_crossing_m, x, 0, 10)

        x = [10, 20, 20, 30]
        self.assertEqualOnAllArrayTypes(number_crossing_m, x, 0, 0)
        self.assertEqualOnAllArrayTypes(number_crossing_m, x, 1, 15)

    def test_c3(self):
        x = [1] * 10
        self.assertAlmostEqualOnAllArrayTypes(c3, x, 1, 0)
        self.assertAlmostEqualOnAllArrayTypes(c3, x, 1, 1)
        self.assertAlmostEqualOnAllArrayTypes(c3, x, 1, 2)
        self.assertAlmostEqualOnAllArrayTypes(c3, x, 1, 3)

        x = [1, 2, -3, 4]
        # 1/2 *(1*2*(-3)+2*(-3)*4) = 1/2 *(-6-24) = -30/2
        self.assertAlmostEqualOnAllArrayTypes(c3, x, -15, 1)
        self.assertAlmostEqualOnAllArrayTypes(c3, x, 0, 2)
        self.assertAlmostEqualOnAllArrayTypes(c3, x, 0, 3)

    def test_binned_entropy(self):
        self.assertAlmostEqualOnAllArrayTypes(binned_entropy, [10] * 100, 0, 10)
        self.assertAlmostEqualOnAllArrayTypes(
            binned_entropy,
            [10] * 10 + [1],
            -(10 / 11 * np.math.log(10 / 11) + 1 / 11 * np.math.log(1 / 11)),
            10,
        )
        self.assertAlmostEqualOnAllArrayTypes(
            binned_entropy,
            [10] * 10 + [1],
            -(10 / 11 * np.math.log(10 / 11) + 1 / 11 * np.math.log(1 / 11)),
            10,
        )
        self.assertAlmostEqualOnAllArrayTypes(
            binned_entropy,
            [10] * 10 + [1],
            -(10 / 11 * np.math.log(10 / 11) + 1 / 11 * np.math.log(1 / 11)),
            100,
        )
        self.assertAlmostEqualOnAllArrayTypes(
            binned_entropy, list(range(10)), -np.math.log(1 / 10), 100
        )
        self.assertAlmostEqualOnAllArrayTypes(
            binned_entropy, list(range(100)), -np.math.log(1 / 2), 2
        )

    def test_sample_entropy(self):
        # "random" list -> large entropy
        ts = [
            1,
            4,
            5,
            1,
            7,
            3,
            1,
            2,
            5,
            8,
            9,
            7,
            3,
            7,
            9,
            5,
            4,
            3,
            9,
            1,
            2,
            3,
            4,
            2,
            9,
            6,
            7,
            4,
            9,
            2,
            9,
            9,
            6,
            5,
            1,
            3,
            8,
            1,
            5,
            3,
            8,
            4,
            1,
            2,
            2,
            1,
            6,
            5,
            3,
            6,
            5,
            4,
            8,
            9,
            6,
            7,
            5,
            3,
            2,
            5,
            4,
            2,
            5,
            1,
            6,
            5,
            3,
            5,
            6,
            7,
            8,
            5,
            2,
            8,
            6,
            3,
            8,
            2,
            7,
            1,
            7,
            3,
            5,
            6,
            2,
            1,
            3,
            7,
            3,
            5,
            3,
            7,
            6,
            7,
            7,
            2,
            3,
            1,
            7,
            8,
        ]
        self.assertAlmostEqualOnAllArrayTypes(sample_entropy, ts, 2.38262780)
        # This is not very complex, so it gives a small value
        ts = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.assertAlmostEqualOnAllArrayTypes(sample_entropy, ts, 0.25131442)
        # however adding a 2 increases complexity
        ts = [1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
        self.assertAlmostEqualOnAllArrayTypes(sample_entropy, ts, 0.74193734)
        # and it does not matter where
        ts = [1, 1, 1, 2, 1, 1, 1, 1, 1, 1]
        self.assertAlmostEqualOnAllArrayTypes(sample_entropy, ts, 0.74193734)
        # negative numbers also work
        ts = [1, -1, 1, -1, 1, -1]
        self.assertAlmostEqualOnAllArrayTypes(sample_entropy, ts, 0.69314718)
        # nan gives nan
        ts = [1, -1, 1, np.nan, 1, -1]
        self.assertIsNanOnAllArrayTypes(sample_entropy, ts)
        # this is not a very "random" list, so it should give a small entropy
        ts = list(range(1000))
        self.assertAlmostEqualOnAllArrayTypes(sample_entropy, ts, 0.0010314596066622707)

    def test_autocorrelation(self):
        self.assertAlmostEqualOnAllArrayTypes(
            autocorrelation, [1, 2, 1, 2, 1, 2], -1, 1
        )
        self.assertAlmostEqualOnAllArrayTypes(autocorrelation, [1, 2, 1, 2, 1, 2], 1, 2)
        self.assertAlmostEqualOnAllArrayTypes(
            autocorrelation, [1, 2, 1, 2, 1, 2], -1, 3
        )
        self.assertAlmostEqualOnAllArrayTypes(autocorrelation, [1, 2, 1, 2, 1, 2], 1, 4)
        self.assertAlmostEqualOnAllArrayTypes(
            autocorrelation, pd.Series([0, 1, 2, 0, 1, 2]), -0.75, 2
        )
        # Autocorrelation lag is larger than length of the time series
        self.assertIsNanOnAllArrayTypes(autocorrelation, [1, 2, 1, 2, 1, 2], 200)
        self.assertIsNanOnAllArrayTypes(autocorrelation, [np.nan], 0)
        self.assertIsNanOnAllArrayTypes(autocorrelation, [], 0)
        # time series with length 1 has no variance, therefore no result for autocorrelation at lag 0
        self.assertIsNanOnAllArrayTypes(autocorrelation, [1], 0)

    def test_quantile(self):
        self.assertAlmostEqualOnAllArrayTypes(
            quantile, [1, 1, 1, 3, 4, 7, 9, 11, 13, 13], 1.0, 0.2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            quantile, [1, 1, 1, 3, 4, 7, 9, 11, 13, 13], 13, 0.9
        )
        self.assertAlmostEqualOnAllArrayTypes(
            quantile, [1, 1, 1, 3, 4, 7, 9, 11, 13, 13], 13, 1.0
        )
        self.assertAlmostEqualOnAllArrayTypes(quantile, [1], 1, 0.5)
        self.assertIsNanOnAllArrayTypes(quantile, [], 0.5)

    def test_mean_abs_change_quantiles(self):

        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            list(range(10)),
            1,
            ql=0.1,
            qh=0.9,
            isabs=True,
            f_agg="mean",
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            list(range(10)),
            0,
            ql=0.15,
            qh=0.18,
            isabs=True,
            f_agg="mean",
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles, [0, 1, 0, 0, 0], 0.5, ql=0, qh=1, isabs=True, f_agg="mean"
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            [0, 1, 0, 0, 0],
            0.5,
            ql=0.1,
            qh=1,
            isabs=True,
            f_agg="mean",
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            [0, 1, 0, 0, 0],
            0,
            ql=0.1,
            qh=0.6,
            isabs=True,
            f_agg="mean",
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles, [0, 1, -9, 0, 0], 5, ql=0, qh=1, isabs=True, f_agg="mean"
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            [0, 1, -9, 0, 0],
            0.5,
            ql=0.1,
            qh=1,
            isabs=True,
            f_agg="mean",
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            [0, 1, -9, 0, 0, 1, 0],
            0.75,
            ql=0.1,
            qh=1,
            isabs=True,
            f_agg="mean",
        )

        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            list(range(10)),
            1,
            ql=0.1,
            qh=0.9,
            isabs=False,
            f_agg="mean",
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            list(range(10)),
            0,
            ql=0.15,
            qh=0.18,
            isabs=False,
            f_agg="mean",
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles, [0, 1, 0, 0, 0], 0, ql=0, qh=1, isabs=False, f_agg="mean"
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            [0, 1, 0, 0, 0],
            0,
            ql=0.1,
            qh=1,
            isabs=False,
            f_agg="mean",
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            [0, 1, 0, 0, 0],
            0,
            ql=0.1,
            qh=0.6,
            isabs=False,
            f_agg="mean",
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles, [0, 1, -9, 0, 0], 0, ql=0, qh=1, isabs=False, f_agg="mean"
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            [0, 1, -9, 0, 0],
            0.5,
            ql=0.1,
            qh=1,
            isabs=False,
            f_agg="mean",
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            [0, 1, -9, 0, 0, 1, 0],
            0.25,
            ql=0.1,
            qh=1,
            isabs=False,
            f_agg="mean",
        )

        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            list(range(10)),
            0,
            ql=0.1,
            qh=0.9,
            isabs=True,
            f_agg="std",
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles, [0, 1, 0, 0, 0], 0.5, ql=0, qh=1, isabs=True, f_agg="std"
        )

        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles,
            list(range(10)),
            0,
            ql=0.1,
            qh=0.9,
            isabs=False,
            f_agg="std",
        )
        self.assertAlmostEqualOnAllArrayTypes(
            change_quantiles, [0, 1, 0, 1, 0], 1, ql=0, qh=1, isabs=False, f_agg="std"
        )

    def test_value_count(self):
        self.assertEqualPandasSeriesWrapper(value_count, [1] * 10, 10, value=1)
        self.assertEqualPandasSeriesWrapper(value_count, list(range(10)), 1, value=0)
        self.assertEqualPandasSeriesWrapper(value_count, [1] * 10, 0, value=0)
        self.assertEqualPandasSeriesWrapper(value_count, [np.NaN, 0, 1] * 3, 3, value=0)
        self.assertEqualPandasSeriesWrapper(
            value_count, [np.NINF, 0, 1] * 3, 3, value=0
        )
        self.assertEqualPandasSeriesWrapper(
            value_count, [np.PINF, 0, 1] * 3, 3, value=0
        )
        self.assertEqualPandasSeriesWrapper(
            value_count, [0.1, 0.2, 0.3] * 3, 3, value=0.2
        )
        self.assertEqualPandasSeriesWrapper(
            value_count, [np.NaN, 0, 1] * 3, 3, value=np.NaN
        )
        self.assertEqualPandasSeriesWrapper(
            value_count, [np.NINF, 0, 1] * 3, 3, value=np.NINF
        )
        self.assertEqualPandasSeriesWrapper(
            value_count, [np.PINF, 0, 1] * 3, 3, value=np.PINF
        )

    def test_range_count(self):
        self.assertEqualPandasSeriesWrapper(range_count, [1] * 10, 0, min=1, max=1)
        self.assertEqualPandasSeriesWrapper(range_count, [1] * 10, 0, min=0.9, max=1)
        self.assertEqualPandasSeriesWrapper(range_count, [1] * 10, 10, min=1, max=1.1)
        self.assertEqualPandasSeriesWrapper(
            range_count, list(range(10)), 9, min=0, max=9
        )
        self.assertEqualPandasSeriesWrapper(
            range_count, list(range(10)), 10, min=0, max=10
        )
        self.assertEqualPandasSeriesWrapper(
            range_count, list(range(0, -10, -1)), 9, min=-10, max=0
        )
        self.assertEqualPandasSeriesWrapper(
            range_count, [np.NaN, np.PINF, np.NINF] + list(range(10)), 10, min=0, max=10
        )

    def test_approximate_entropy(self):
        self.assertEqualOnAllArrayTypes(approximate_entropy, [1], 0, m=2, r=0.5)
        self.assertEqualOnAllArrayTypes(approximate_entropy, [1, 2], 0, m=2, r=0.5)
        self.assertEqualOnAllArrayTypes(approximate_entropy, [1, 2, 3], 0, m=2, r=0.5)
        self.assertEqualOnAllArrayTypes(approximate_entropy, [1, 2, 3], 0, m=2, r=0.5)
        self.assertAlmostEqualOnAllArrayTypes(
            approximate_entropy, [12, 13, 15, 16, 17] * 10, 0.282456191, m=2, r=0.9
        )
        self.assertRaises(
            ValueError, approximate_entropy, x=[12, 13, 15, 16, 17] * 10, m=2, r=-0.5
        )

    def test_absolute_maximum(self):
        self.assertEqualOnAllArrayTypes(absolute_maximum, [-5, 0, 1], 5)
        self.assertEqualOnAllArrayTypes(absolute_maximum, [0], 0)
        self.assertIsNanOnAllArrayTypes(absolute_maximum, [])

    def test_max_langevin_fixed_point(self):
        """
        Estimating the intrinsic velocity of a dissipative soliton
        """
        default_params = {"m": 3, "r": 30}

        # active Brownian motion
        ds = velocity(tau=3.8, delta_t=0.05, R=3e-4, seed=0)
        v = ds.simulate(100000, v0=np.zeros(1))
        v0 = max_langevin_fixed_point(v[:, 0], **default_params)
        self.assertLess(abs(ds.deterministic - v0), 0.001)

        # Brownian motion
        ds = velocity(tau=2.0 / 0.3 - 3.8, delta_t=0.05, R=3e-4, seed=0)
        v = ds.simulate(10000, v0=np.zeros(1))
        v0 = max_langevin_fixed_point(v[:, 0], **default_params)
        self.assertLess(v0, 0.001)

    def test_linear_trend(self):
        # check linear up trend
        x = range(10)
        param = [
            {"attr": "pvalue"},
            {"attr": "rvalue"},
            {"attr": "intercept"},
            {"attr": "slope"},
            {"attr": "stderr"},
        ]
        res = linear_trend(x, param)

        res = pd.Series(dict(res))

        expected_index = [
            'attr_"pvalue"',
            'attr_"intercept"',
            'attr_"rvalue"',
            'attr_"slope"',
            'attr_"stderr"',
        ]

        self.assertEqual(len(res), 5)
        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res['attr_"pvalue"'], 0)
        self.assertAlmostEqual(res['attr_"stderr"'], 0)
        self.assertAlmostEqual(res['attr_"intercept"'], 0)
        self.assertAlmostEqual(res['attr_"slope"'], 1.0)

        # check p value for random trend
        np.random.seed(42)
        x = np.random.uniform(size=100)
        param = [{"attr": "rvalue"}]
        res = linear_trend(x, param)

        res = pd.Series(dict(res))

        self.assertLess(abs(res['attr_"rvalue"']), 0.1)

        # check slope and intercept decreasing trend with intercept
        x = [42 - 2 * x for x in range(10)]
        param = [{"attr": "intercept"}, {"attr": "slope"}]
        res = linear_trend(x, param)

        res = pd.Series(dict(res))

        self.assertAlmostEqual(res['attr_"intercept"'], 42)
        self.assertAlmostEqual(res['attr_"slope"'], -2)

    def test__aggregate_on_chunks(self):
        self.assertListEqual(
            _aggregate_on_chunks(x=pd.Series([0, 1, 2, 3]), f_agg="max", chunk_len=2),
            [1, 3],
        )
        self.assertListEqual(
            _aggregate_on_chunks(x=pd.Series([1, 1, 3, 3]), f_agg="max", chunk_len=2),
            [1, 3],
        )

        self.assertListEqual(
            _aggregate_on_chunks(x=pd.Series([0, 1, 2, 3]), f_agg="min", chunk_len=2),
            [0, 2],
        )
        self.assertListEqual(
            _aggregate_on_chunks(
                x=pd.Series([0, 1, 2, 3, 5]), f_agg="min", chunk_len=2
            ),
            [0, 2, 5],
        )

        self.assertListEqual(
            _aggregate_on_chunks(x=pd.Series([0, 1, 2, 3]), f_agg="mean", chunk_len=2),
            [0.5, 2.5],
        )
        self.assertListEqual(
            _aggregate_on_chunks(
                x=pd.Series([0, 1, 0, 4, 5]), f_agg="mean", chunk_len=2
            ),
            [0.5, 2, 5],
        )
        self.assertListEqual(
            _aggregate_on_chunks(
                x=pd.Series([0, 1, 0, 4, 5]), f_agg="mean", chunk_len=3
            ),
            [1 / 3, 4.5],
        )

        self.assertListEqual(
            _aggregate_on_chunks(
                x=pd.Series([0, 1, 2, 3, 5, -2]), f_agg="median", chunk_len=2
            ),
            [0.5, 2.5, 1.5],
        )
        self.assertListEqual(
            _aggregate_on_chunks(
                x=pd.Series([-10, 5, 3, -3, 4, -6]), f_agg="median", chunk_len=3
            ),
            [3, -3],
        )
        self.assertListEqual(
            _aggregate_on_chunks(
                x=pd.Series([0, 1, 2, np.NaN, 5]), f_agg="median", chunk_len=2
            ),
            [0.5, 2, 5],
        )

    def test_agg_linear_trend(self):
        x = pd.Series(range(9), index=range(9))
        param = [
            {"attr": "intercept", "chunk_len": 3, "f_agg": "max"},
            {"attr": "slope", "chunk_len": 3, "f_agg": "max"},
            {"attr": "intercept", "chunk_len": 3, "f_agg": "min"},
            {"attr": "slope", "chunk_len": 3, "f_agg": "min"},
            {"attr": "intercept", "chunk_len": 3, "f_agg": "mean"},
            {"attr": "slope", "chunk_len": 3, "f_agg": "mean"},
            {"attr": "intercept", "chunk_len": 3, "f_agg": "median"},
            {"attr": "slope", "chunk_len": 3, "f_agg": "median"},
        ]
        expected_index = [
            'attr_"intercept"__chunk_len_3__f_agg_"max"',
            'attr_"slope"__chunk_len_3__f_agg_"max"',
            'attr_"intercept"__chunk_len_3__f_agg_"min"',
            'attr_"slope"__chunk_len_3__f_agg_"min"',
            'attr_"intercept"__chunk_len_3__f_agg_"mean"',
            'attr_"slope"__chunk_len_3__f_agg_"mean"',
            'attr_"intercept"__chunk_len_3__f_agg_"median"',
            'attr_"slope"__chunk_len_3__f_agg_"median"',
        ]

        res = agg_linear_trend(x=x, param=param)

        res = pd.Series(dict(res))
        self.assertEqual(len(res), 8)
        self.maxDiff = 2000
        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res['attr_"intercept"__chunk_len_3__f_agg_"max"'], 2)
        self.assertAlmostEqual(res['attr_"slope"__chunk_len_3__f_agg_"max"'], 3)
        self.assertAlmostEqual(res['attr_"intercept"__chunk_len_3__f_agg_"min"'], 0)
        self.assertAlmostEqual(res['attr_"slope"__chunk_len_3__f_agg_"min"'], 3)
        self.assertAlmostEqual(res['attr_"intercept"__chunk_len_3__f_agg_"mean"'], 1)
        self.assertAlmostEqual(res['attr_"slope"__chunk_len_3__f_agg_"mean"'], 3)
        self.assertAlmostEqual(res['attr_"intercept"__chunk_len_3__f_agg_"median"'], 1)
        self.assertAlmostEqual(res['attr_"slope"__chunk_len_3__f_agg_"median"'], 3)

        x = pd.Series([np.NaN, np.NaN, np.NaN, -3, -3, -3])
        res = agg_linear_trend(x=x, param=param)

        res = pd.Series(dict(res))

        self.assertIsNaN(res['attr_"intercept"__chunk_len_3__f_agg_"max"'])
        self.assertIsNaN(res['attr_"slope"__chunk_len_3__f_agg_"max"'])
        self.assertIsNaN(res['attr_"intercept"__chunk_len_3__f_agg_"min"'])
        self.assertIsNaN(res['attr_"slope"__chunk_len_3__f_agg_"min"'])
        self.assertIsNaN(res['attr_"intercept"__chunk_len_3__f_agg_"mean"'])
        self.assertIsNaN(res['attr_"slope"__chunk_len_3__f_agg_"mean"'])
        self.assertIsNaN(res['attr_"intercept"__chunk_len_3__f_agg_"median"'])
        self.assertIsNaN(res['attr_"slope"__chunk_len_3__f_agg_"median"'])

        x = pd.Series([np.NaN, np.NaN, -3, -3, -3, -3])
        res = agg_linear_trend(x=x, param=param)

        res = pd.Series(dict(res))

        self.assertAlmostEqual(res['attr_"intercept"__chunk_len_3__f_agg_"max"'], -3)
        self.assertAlmostEqual(res['attr_"slope"__chunk_len_3__f_agg_"max"'], 0)
        self.assertAlmostEqual(res['attr_"intercept"__chunk_len_3__f_agg_"min"'], -3)
        self.assertAlmostEqual(res['attr_"slope"__chunk_len_3__f_agg_"min"'], 0)
        self.assertAlmostEqual(res['attr_"intercept"__chunk_len_3__f_agg_"mean"'], -3)
        self.assertAlmostEqual(res['attr_"slope"__chunk_len_3__f_agg_"mean"'], 0)
        self.assertAlmostEqual(res['attr_"intercept"__chunk_len_3__f_agg_"median"'], -3)
        self.assertAlmostEqual(res['attr_"slope"__chunk_len_3__f_agg_"median"'], 0)

    def test_energy_ratio_by_chunks(self):
        x = pd.Series(range(90), index=range(90))
        param = [{"num_segments": 6, "segment_focus": i} for i in range(6)]
        output = energy_ratio_by_chunks(x=x, param=param)

        self.assertAlmostEqual(output[0][1], 0.0043, places=3)
        self.assertAlmostEqual(output[1][1], 0.0316, places=3)
        self.assertAlmostEqual(output[2][1], 0.0871, places=3)
        self.assertAlmostEqual(output[3][1], 0.1709, places=3)
        self.assertAlmostEqual(output[4][1], 0.2829, places=3)
        self.assertAlmostEqual(output[5][1], 0.4232, places=3)

        # Sum of the ratios should be 1.0
        sum = 0.0
        for name, dat in output:
            sum = sum + dat
        self.assertAlmostEqual(sum, 1.0)

        x = pd.Series(1, index=range(10))
        param = [{"num_segments": 3, "segment_focus": i} for i in range(3)]
        output = energy_ratio_by_chunks(x=x, param=param)

        self.assertAlmostEqual(output[0][1], 0.4, places=3)
        self.assertAlmostEqual(output[1][1], 0.3, places=3)
        self.assertAlmostEqual(output[2][1], 0.3, places=3)

        # Sum of the ratios should be 1.0
        sum = 0.0
        for name, dat in output:
            sum = sum + dat
        self.assertAlmostEqual(sum, 1.0)

        x = pd.Series(0, index=range(10))
        param = [{"num_segments": 3, "segment_focus": i} for i in range(3)]
        output = energy_ratio_by_chunks(x=x, param=param)
        self.assertIsNaN(output[0][1])
        self.assertIsNaN(output[1][1])
        self.assertIsNaN(output[2][1])

    def test_linear_trend_timewise_hours(self):
        """Test linear_trend_timewise function with hour intervals."""
        x = pd.Series(
            [0, 1, 3, 6],
            index=pd.DatetimeIndex(
                [
                    "2018-01-01 04:00:00",
                    "2018-01-01 05:00:00",
                    "2018-01-01 07:00:00",
                    "2018-01-01 10:00:00",
                ]
            ),
        )

        param = [
            {"attr": "pvalue"},
            {"attr": "rvalue"},
            {"attr": "intercept"},
            {"attr": "slope"},
            {"attr": "stderr"},
        ]
        res = linear_trend_timewise(x, param)

        res = pd.Series(dict(res))

        expected_index = [
            'attr_"pvalue"',
            'attr_"intercept"',
            'attr_"rvalue"',
            'attr_"slope"',
            'attr_"stderr"',
        ]

        self.assertEqual(len(res), 5)
        self.assertCountEqual(list(res.index), expected_index)
        self.assertAlmostEqual(res['attr_"pvalue"'], 0, places=3)
        self.assertAlmostEqual(res['attr_"stderr"'], 0, places=3)
        self.assertAlmostEqual(res['attr_"intercept"'], 0, places=3)
        self.assertAlmostEqual(res['attr_"slope"'], 1.0, places=3)

    def test_linear_trend_timewise_days(self):
        """Test linear_trend_timewise function with day intervals."""
        # Try with different days
        x = pd.Series(
            [0, 24, 48, 72],
            index=pd.DatetimeIndex(
                [
                    "2018-01-01 04:00:00",
                    "2018-01-02 04:00:00",
                    "2018-01-03 04:00:00",
                    "2018-01-04 04:00:00",
                ]
            ),
        )

        param = [
            {"attr": "pvalue"},
            {"attr": "rvalue"},
            {"attr": "intercept"},
            {"attr": "slope"},
            {"attr": "stderr"},
        ]
        res = linear_trend_timewise(x, param)

        res = pd.Series(dict(res))

        self.assertAlmostEqual(res['attr_"pvalue"'], 0, places=3)
        self.assertAlmostEqual(res['attr_"stderr"'], 0, places=3)
        self.assertAlmostEqual(res['attr_"intercept"'], 0, places=3)
        self.assertAlmostEqual(res['attr_"slope"'], 1.0, places=3)

    def test_linear_trend_timewise_seconds(self):
        """Test linear_trend_timewise function with second intervals."""
        # Try with different days
        x = pd.Series(
            [0, 1 / float(3600), 2 / float(3600), 3 / float(3600)],
            index=pd.DatetimeIndex(
                [
                    "2018-01-01 04:00:01",
                    "2018-01-01 04:00:02",
                    "2018-01-01 04:00:03",
                    "2018-01-01 04:00:04",
                ]
            ),
        )

        param = [
            {"attr": "pvalue"},
            {"attr": "rvalue"},
            {"attr": "intercept"},
            {"attr": "slope"},
            {"attr": "stderr"},
        ]
        res = linear_trend_timewise(x, param)

        res = pd.Series(dict(res))

        self.assertAlmostEqual(res['attr_"pvalue"'], 0, places=3)
        self.assertAlmostEqual(res['attr_"stderr"'], 0, places=3)
        self.assertAlmostEqual(res['attr_"intercept"'], 0, places=3)
        self.assertAlmostEqual(res['attr_"slope"'], 1.0, places=3)

    def test_linear_trend_timewise_years(self):
        """Test linear_trend_timewise function with year intervals."""
        # Try with different days
        x = pd.Series(
            [
                0,
                365 * 24,
                365 * 48,
                365 * 72 + 24,
            ],  # Add 24 to the last one since it's a leap year
            index=pd.DatetimeIndex(
                [
                    "2018-01-01 04:00:00",
                    "2019-01-01 04:00:00",
                    "2020-01-01 04:00:00",
                    "2021-01-01 04:00:00",
                ]
            ),
        )

        param = [
            {"attr": "pvalue"},
            {"attr": "rvalue"},
            {"attr": "intercept"},
            {"attr": "slope"},
            {"attr": "stderr"},
        ]
        res = linear_trend_timewise(x, param)

        res = pd.Series(dict(res))

        self.assertAlmostEqual(res['attr_"pvalue"'], 0, places=3)
        self.assertAlmostEqual(res['attr_"stderr"'], 0, places=3)
        self.assertAlmostEqual(res['attr_"intercept"'], 0, places=3)
        self.assertAlmostEqual(res['attr_"slope"'], 1.0, places=3)

    def test_change_quantiles(self):
        """Test change_quantiles function when changing from `sum` to `np.sum`."""
        np.random.seed(0)
        res = change_quantiles(np.random.rand(10000) * 1000, 0.1, 0.2, False, "mean")
        self.assertAlmostEqual(res, -0.9443846621365727)

    def test_count_above(self):
        self.assertEqualPandasSeriesWrapper(count_above, [1] * 10, 1, t=1)
        self.assertEqualPandasSeriesWrapper(count_above, list(range(10)), 1, t=0)
        self.assertEqualPandasSeriesWrapper(count_above, list(range(10)), 0.5, t=5)
        self.assertEqualPandasSeriesWrapper(
            count_above, [0.1, 0.2, 0.3] * 3, 2 / 3, t=0.2
        )
        self.assertEqualPandasSeriesWrapper(count_above, [np.NaN, 0, 1] * 3, 2 / 3, t=0)
        self.assertEqualPandasSeriesWrapper(
            count_above, [np.NINF, 0, 1] * 3, 2 / 3, t=0
        )
        self.assertEqualPandasSeriesWrapper(count_above, [np.PINF, 0, 1] * 3, 1, t=0)
        self.assertEqualPandasSeriesWrapper(
            count_above, [np.NaN, 0, 1] * 3, 0, t=np.NaN
        )
        self.assertEqualPandasSeriesWrapper(
            count_above, [np.NINF, 0, np.PINF] * 3, 1, t=np.NINF
        )
        self.assertEqualPandasSeriesWrapper(
            count_above, [np.PINF, 0, 1] * 3, 1 / 3, t=np.PINF
        )

    def test_count_below(self):
        self.assertEqualPandasSeriesWrapper(count_below, [1] * 10, 1, t=1)
        self.assertEqualPandasSeriesWrapper(count_below, list(range(10)), 1 / 10, t=0)
        self.assertEqualPandasSeriesWrapper(count_below, list(range(10)), 6 / 10, t=5)
        self.assertEqualPandasSeriesWrapper(
            count_below, [0.1, 0.2, 0.3] * 3, 2 / 3, t=0.2
        )
        self.assertEqualPandasSeriesWrapper(count_below, [np.NaN, 0, 1] * 3, 1 / 3, t=0)
        self.assertEqualPandasSeriesWrapper(
            count_below, [np.NINF, 0, 1] * 3, 2 / 3, t=0
        )
        self.assertEqualPandasSeriesWrapper(
            count_below, [np.PINF, 0, 1] * 3, 1 / 3, t=0
        )
        self.assertEqualPandasSeriesWrapper(
            count_below, [np.NaN, 0, 1] * 3, 0, t=np.NaN
        )
        self.assertEqualPandasSeriesWrapper(
            count_below, [np.NINF, 0, np.PINF] * 3, 1 / 3, t=np.NINF
        )
        self.assertEqualPandasSeriesWrapper(
            count_below, [np.PINF, 0, 1] * 3, 1, t=np.PINF
        )

    def test_benford_correlation(self):
        # A test with list of random values
        np.random.seed(42)
        random_list = np.random.uniform(size=100)

        # Fibonacci series is known to match the Newcomb-Benford's Distribution
        fibonacci_list = [0, 1]
        for i in range(2, 200):
            fibonacci_list.append(fibonacci_list[i - 1] + fibonacci_list[i - 2])

        # A list of equally distributed digits (returns NaN)
        equal_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        # A list containing NaN
        list_with_nan = [
            1.354,
            0.058,
            0.055,
            0.99,
            3.15,
            np.nan,
            0.3,
            2.3,
            0,
            0.59,
            0.74,
        ]

        self.assertAlmostEqual(benford_correlation(random_list), 0.39458056)
        self.assertAlmostEqual(benford_correlation(fibonacci_list), 0.998003988)
        self.assertAlmostEqual(benford_correlation(list_with_nan), 0.10357511)
        self.assertIsNaN(benford_correlation(equal_list))

    def test_query_similarity_count(self):
        np.random.seed(42)
        query = np.random.uniform(size=10)
        threshold = 3.0
        x = np.random.uniform(size=100)

        # z-normalized Euclidean distances

        param = [{"query": query}]
        self.assertAlmostEqual(query_similarity_count(x, param=param)[0][1], 0.0)

        param = [{"query": query, "threshold": threshold}]
        self.assertAlmostEqual(query_similarity_count(x, param=param)[0][1], 6.0)

        # non-normalized Euclidean distances

        param = [{"query": query, "normalize": False}]
        self.assertAlmostEqual(query_similarity_count(x, param=param)[0][1], 0.0)

        param = [{"query": query, "threshold": threshold, "normalize": False}]
        self.assertAlmostEqual(query_similarity_count(x, param=param)[0][1], 91.0)

    def test_matrix_profile_window(self):
        # Test matrix profile output with specified window
        np.random.seed(9999)
        ts = np.random.uniform(size=2 ** 10)
        w = 2 ** 5
        subq = ts[0:w]
        ts[0:w] = subq
        ts[w + 100 : w + 100 + w] = subq
        param = [
            {"threshold": 0.98, "windows": 36, "feature": "min"},
            {"threshold": 0.98, "windows": 36, "feature": "max"},
            {"threshold": 0.98, "windows": 36, "feature": "mean"},
            {"threshold": 0.98, "windows": 36, "feature": "median"},
            {"threshold": 0.98, "windows": 36, "feature": "25"},
            {"threshold": 0.98, "windows": 36, "feature": "75"},
        ]

        self.assertAlmostEqual(matrix_profile(ts, param=param)[0][1], 2.825786727580335)

    def test_matrix_profile_no_window(self):
        # Test matrix profile output with no window specified
        np.random.seed(9999)
        ts = np.random.uniform(size=2 ** 10)
        w = 2 ** 5
        subq = ts[0:w]
        ts[0:w] = subq
        ts[w + 100 : w + 100 + w] = subq

        param = [
            {"threshold": 0.98, "feature": "min"},
            {"threshold": 0.98, "feature": "max"},
            {"threshold": 0.98, "feature": "mean"},
            {"threshold": 0.98, "feature": "median"},
            {"threshold": 0.98, "feature": "25"},
            {"threshold": 0.98, "feature": "75"},
        ]

        # Test matrix profile output with no window specified
        self.assertAlmostEqual(matrix_profile(ts, param=param)[0][1], 2.825786727580335)

    def test_matrix_profile_nan(self):
        # Test matrix profile of NaNs (NaN output)
        ts = np.random.uniform(size=2 ** 6)
        ts[:] = np.nan

        param = [
            {"threshold": 0.98, "windows": None, "feature": "min"},
            {"threshold": 0.98, "windows": None, "feature": "max"},
            {"threshold": 0.98, "windows": None, "feature": "mean"},
            {"threshold": 0.98, "windows": None, "feature": "median"},
            {"threshold": 0.98, "windows": None, "feature": "25"},
            {"threshold": 0.98, "windows": None, "feature": "75"},
        ]

        self.assertTrue(np.isnan(matrix_profile(ts, param=param)[0][1]))


class FriedrichTestCase(TestCase):
    def test_estimate_friedrich_coefficients(self):
        """
        Estimate friedrich coefficients
        """
        default_params = {"m": 3, "r": 30}

        # active Brownian motion
        ds = velocity(tau=3.8, delta_t=0.05, R=3e-4, seed=0)
        v = ds.simulate(10000, v0=np.zeros(1))
        coeff = _estimate_friedrich_coefficients(v[:, 0], **default_params)
        self.assertLess(abs(coeff[-1]), 0.0001)

        # Brownian motion
        ds = velocity(tau=2.0 / 0.3 - 3.8, delta_t=0.05, R=3e-4, seed=0)
        v = ds.simulate(10000, v0=np.zeros(1))
        coeff = _estimate_friedrich_coefficients(v[:, 0], **default_params)
        self.assertLess(abs(coeff[-1]), 0.0001)

    def test_friedrich_coefficients(self):
        # Test binning error returns vector of NaNs
        param = [{"coeff": coeff, "m": 2, "r": 30} for coeff in range(4)]
        x = np.zeros(100)
        res = pd.Series(dict(friedrich_coefficients(x, param)))

        expected_index = [
            "coeff_0__m_2__r_30",
            "coeff_1__m_2__r_30",
            "coeff_2__m_2__r_30",
            "coeff_3__m_2__r_30",
        ]
        self.assertCountEqual(list(res.index), expected_index)
        self.assertTrue(np.sum(np.isnan(res)), 3)

    def test_friedrich_number_of_returned_features_is_equal_to_number_of_parameters(
        self,
    ):
        """unit test for issue 501"""
        param = [
            {"m": 3, "r": 5, "coeff": 2},
            {"m": 3, "r": 5, "coeff": 3},
            {"m": 3, "r": 2, "coeff": 3},
        ]
        x = np.zeros(100)
        res = pd.Series(dict(friedrich_coefficients(x, param)))

        expected_index = ["coeff_2__m_3__r_5", "coeff_3__m_3__r_5", "coeff_3__m_3__r_2"]
        self.assertCountEqual(list(res.index), expected_index)
        self.assertTrue(np.sum(np.isnan(res)), 3)

    def test_friedrich_equal_to_snapshot(self):
        param = [{"coeff": coeff, "m": 2, "r": 30} for coeff in range(4)]
        x = np.array(
            [
                -0.53,
                -0.61,
                -1.26,
                -0.88,
                -0.34,
                0.58,
                2.86,
                -0.47,
                0.78,
                -0.45,
                -0.27,
                0.43,
                1.72,
                0.26,
                1.02,
                -0.09,
                0.65,
                1.49,
                -0.95,
                -1.02,
                -0.64,
                -1.63,
                -0.71,
                -0.43,
                -1.69,
                0.05,
                1.58,
                1.1,
                0.55,
                -1.02,
            ]
        )

        res = pd.Series(dict(friedrich_coefficients(x, param)))

        self.assertAlmostEqual(res["coeff_0__m_2__r_30"], -0.24536975738843042)
        self.assertAlmostEqual(res["coeff_1__m_2__r_30"], -0.533309548662685)
        self.assertAlmostEqual(res["coeff_2__m_2__r_30"], 0.2759399238199404)
