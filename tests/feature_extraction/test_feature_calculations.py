# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from __future__ import absolute_import, division
from __future__ import print_function
from builtins import range
from random import shuffle
from unittest import TestCase
from tsfresh.feature_extraction.feature_calculators import *
from tsfresh.feature_extraction.feature_calculators import _get_length_sequences_where
import six
import math

class FeatureCalculationTestCase(TestCase):

    def assertEqualOnAllArrayTypes(self, f, input_to_f, result, *args, **kwargs):
        self.assertEqual(f(input_to_f, *args, **kwargs), result,
                         msg="Not equal for lists: %s != %s" % (f(input_to_f, *args, **kwargs), result))
        self.assertEqual(f(np.array(input_to_f), *args, **kwargs), result,
                         msg="Not equal for numpy.arrays: %s != %s" % (f(np.array(input_to_f), *args, **kwargs), result))
        self.assertEqual(f(pd.Series(input_to_f), *args, **kwargs), result,
                         msg="Not equal for pandas.Series: %s != %s" % (f(pd.Series(input_to_f), *args, **kwargs), result))

    def assertTrueOnAllArrayTypes(self, f, input_to_f, *args, **kwargs):
        self.assertTrue(f(input_to_f, *args, **kwargs), msg="Not true for lists")
        self.assertTrue(f(np.array(input_to_f), *args, **kwargs), msg="Not true for numpy.arrays")
        self.assertTrue(f(pd.Series(input_to_f), *args, **kwargs), msg="Not true for pandas.Series")

    def assertFalseOnAllArrayTypes(self, f, input_to_f, *args, **kwargs):
        self.assertFalse(f(input_to_f, *args, **kwargs), msg="Not false for lists")
        self.assertFalse(f(np.array(input_to_f), *args, **kwargs), msg="Not false for numpy.arrays")
        self.assertFalse(f(pd.Series(input_to_f), *args, **kwargs), msg="Not false for pandas.Series")

    def assertAlmostEqualOnAllArrayTypes(self, f, input_t_f, result, *args, **kwargs):
        self.assertAlmostEqual(f(input_t_f, *args, **kwargs), result,
                               msg="Not almost equal for lists: %s != %s" % (f(input_t_f, *args, **kwargs), result))
        self.assertAlmostEqual(f(np.array(input_t_f), *args, **kwargs), result,
                               msg="Not almost equal for np.arrays: %s != %s" % (
                                   f(np.array(input_t_f), *args, **kwargs), result))
        self.assertAlmostEqual(f(pd.Series(input_t_f), *args, **kwargs), result,
                               msg="Not almost equal for pd.Series: %s != %s" % (
                                   f(pd.Series(input_t_f), *args, **kwargs), result))

    def assertIsNanOnAllArrayTypes(self, f, input_to_f, *args, **kwargs):
        self.assertTrue(np.isnan(f(input_to_f, *args, **kwargs)), msg="Not NaN for lists")
        self.assertTrue(np.isnan(f(np.array(input_to_f), *args, **kwargs)), msg="Not NaN for numpy.arrays")
        self.assertTrue(np.isnan(f(pd.Series(input_to_f), *args, **kwargs)), msg="Not NaN for pandas.Series")

    def assertEqualPandasSeriesWrapper(self, f, input_to_f, result, *args, **kwargs):
        self.assertEqual(f(pd.Series(input_to_f), *args, **kwargs), result,
                         msg="Not equal for pandas.Series: %s != %s" % (
                         f(pd.Series(input_to_f), *args, **kwargs), result))

    def test_catch_Numbers(self):
        def f(x, y):
            return [x, y]
        fdeco = not_apply_to_raw_numbers(f)
        self.assertEqual(fdeco(3, 5), 0)
        self.assertEqual(fdeco([], 5), [[], 5])
        self.assertEqual(fdeco(np.NaN, 10), 0)

    def test___get_length_sequences_where(self):
        self.assertEqualOnAllArrayTypes(_get_length_sequences_where, [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
                                        [1, 3, 1, 2])
        self.assertEqualOnAllArrayTypes(_get_length_sequences_where,
                                        [0, True, 0, 0, True, True, True, 0, 0, True, 0, True, True],
                                        [1, 3, 1, 2])
        self.assertEqualOnAllArrayTypes(_get_length_sequences_where,
                                        [0, True, 0, 0, 1, True, 1, 0, 0, True, 0, 1, True], [1, 3, 1, 2])
        self.assertEqualOnAllArrayTypes(_get_length_sequences_where, [0] * 10, [0])
        self.assertEqualOnAllArrayTypes(_get_length_sequences_where, [], [0])

    def test_variance_larger_than_standard_deviation(self):
        self.assertFalseOnAllArrayTypes(variance_larger_than_standard_deviation, [-1, -1, 1, 1, 1])
        self.assertTrueOnAllArrayTypes(variance_larger_than_standard_deviation, [-1, -1, 1, 1, 2])

    def test_large_standard_deviation(self):
        self.assertFalseOnAllArrayTypes(large_standard_deviation, [1, 1, 1, 1], r=0)
        self.assertFalseOnAllArrayTypes(large_standard_deviation, [1, 1, 1, 1], r=0)
        self.assertTrueOnAllArrayTypes(large_standard_deviation, [-1, -1, 1, 1], r=0)
        self.assertTrueOnAllArrayTypes(large_standard_deviation, [-1, -1, 1, 1], r=0.25)
        self.assertTrueOnAllArrayTypes(large_standard_deviation, [-1, -1, 1, 1], r=0.3)
        self.assertFalseOnAllArrayTypes(large_standard_deviation, [-1, -1, 1, 1], r=0.5)

    def test_symmetry_looking(self):
        self.assertTrueOnAllArrayTypes(symmetry_looking, [-1, -1, 1, 1], r=0.05)
        self.assertTrueOnAllArrayTypes(symmetry_looking, [-1, -1, 1, 1], r=0.75)
        self.assertFalseOnAllArrayTypes(symmetry_looking, [-1, -1, 1, 1], r=0)
        self.assertFalseOnAllArrayTypes(symmetry_looking, [-1, -1, -1, -1, 1], r=0.05)
        self.assertTrueOnAllArrayTypes(symmetry_looking, [-2, -2, -2, -1, -1, -1], r=0.05)
        self.assertTrueOnAllArrayTypes(symmetry_looking, [-0.9, -0.900001], r=0.05)

    def test_has_duplicate_max(self):
        self.assertTrueOnAllArrayTypes(has_duplicate_max, [2.1, 0, 0, 2.1, 1.1])
        self.assertFalseOnAllArrayTypes(has_duplicate_max, np.array([2.1, 0, 0, 2, 1.1]))
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

    def test_large_number_of_peaks(self):
        x = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
        self.assertTrueOnAllArrayTypes(large_number_of_peaks, x, 1)
        self.assertTrueOnAllArrayTypes(large_number_of_peaks, x, 2)
        self.assertFalseOnAllArrayTypes(large_number_of_peaks, x, 3)
        self.assertFalseOnAllArrayTypes(large_number_of_peaks, x, 4)
        self.assertFalseOnAllArrayTypes(large_number_of_peaks, x, 5)
        self.assertFalseOnAllArrayTypes(large_number_of_peaks, x, 6)

    def test_mean_autocorrelation(self):

        x = [1, 1, 1, 1, 1, 1, 1]
        self.assertAlmostEqualOnAllArrayTypes(mean_autocorrelation, x, 0)

        x = [1, 2, -3]
        expected_res = 1/np.var(x) * (1*2+2*(-3)-3/2)/2
        self.assertAlmostEqualOnAllArrayTypes(mean_autocorrelation, x, expected_res)

    def test_augmented_dickey_fuller(self):
        pass
        # todo: add unit test

    def test_abs_energy(self):
        self.assertEqualOnAllArrayTypes(abs_energy, [1, 1, 1], 3)
        self.assertEqualOnAllArrayTypes(abs_energy, [1, 2, 3], 14)
        self.assertEqualOnAllArrayTypes(abs_energy, [-1, 2, -3], 14)
        self.assertAlmostEqualOnAllArrayTypes(abs_energy, [-1, 1.3], 2.69)
        self.assertEqualOnAllArrayTypes(abs_energy, [1], 1)

    def test_mean_abs_change(self):
        self.assertEqualOnAllArrayTypes(mean_abs_change, [-2, 2, 5], 3.5)
        self.assertEqualOnAllArrayTypes(mean_abs_change, [1, 2, -1], 2)

    def test_mean_change(self):
        self.assertEqualOnAllArrayTypes(mean_change, [-2, 2, 5], 3.5)
        self.assertEqualOnAllArrayTypes(mean_change, [1, 2, -1], -1)

    def test_mean_second_derivate_central(self):
        self.assertEqualOnAllArrayTypes(mean_second_derivate_central, list(range(10)), 0)
        self.assertEqualOnAllArrayTypes(mean_second_derivate_central, [1, 3, 5], 0)
        self.assertEqualOnAllArrayTypes(mean_second_derivate_central, [1, 3, 7, -3], -3)

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
        self.assertAlmostEqualOnAllArrayTypes(standard_deviation, [1, 2, -2, -1], 1.58113883008)
        self.assertIsNanOnAllArrayTypes(standard_deviation, [])

    def test_variance(self):
        self.assertAlmostEqualOnAllArrayTypes(variance, [1, 1, -1, -1], 1)
        self.assertAlmostEqualOnAllArrayTypes(variance, [1, 2, -2, -1], 2.5)
        self.assertIsNanOnAllArrayTypes(variance, [])

    def test_skewness(self):
        self.assertEqualOnAllArrayTypes(skewness, [1, 1, 1, 2, 2, 2], 0)
        self.assertAlmostEqualOnAllArrayTypes(skewness, [1, 1, 1, 2, 2], 0.6085806194501855)
        self.assertEqualOnAllArrayTypes(skewness, [1, 1, 1], 0)
        self.assertIsNanOnAllArrayTypes(skewness, [1, 1])

    def test_kurtosis(self):
        self.assertAlmostEqualOnAllArrayTypes(kurtosis, [1, 1, 1, 2, 2], -3.333333333333333)
        self.assertAlmostEqualOnAllArrayTypes(kurtosis, [1, 1, 1, 1], 0)
        self.assertIsNanOnAllArrayTypes(kurtosis, [1, 1, 1])

    def test_absolute_sum_of_changes(self):
        self.assertEqualOnAllArrayTypes(absolute_sum_of_changes, [1, 1, 1, 1, 2, 1], 2)
        self.assertEqualOnAllArrayTypes(absolute_sum_of_changes, [1, -1, 1, -1], 6)
        self.assertEqualOnAllArrayTypes(absolute_sum_of_changes, [1], 0)
        self.assertEqualOnAllArrayTypes(absolute_sum_of_changes, [], 0)

    def test_longest_strike_below_mean(self):
        self.assertEqualOnAllArrayTypes(longest_strike_below_mean, [1, 2, 1, 1, 1, 2, 2, 2], 3)
        self.assertEqualOnAllArrayTypes(longest_strike_below_mean, [1, 2, 1], 1)
        self.assertEqualOnAllArrayTypes(longest_strike_below_mean, [], 0)

    def test_longest_strike_above_mean(self):
        self.assertEqualOnAllArrayTypes(longest_strike_above_mean, [1, 2, 1, 2, 1, 2, 2, 1], 2)
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
        self.assertAlmostEqualOnAllArrayTypes(last_location_of_maximum, [1, 2, 1, 2, 1], 0.8)
        self.assertAlmostEqualOnAllArrayTypes(last_location_of_maximum, [1, 2, 1, 1, 2], 1.0)
        self.assertAlmostEqualOnAllArrayTypes(last_location_of_maximum, [2, 1, 1, 1, 1], 0.2)
        self.assertAlmostEqualOnAllArrayTypes(last_location_of_maximum, [1, 1, 1, 1, 1], 1.0)
        self.assertAlmostEqualOnAllArrayTypes(last_location_of_maximum, [1], 1.0)
        self.assertIsNanOnAllArrayTypes(last_location_of_maximum, [])

    def test_first_location_of_maximum(self):
        self.assertAlmostEqualOnAllArrayTypes(first_location_of_maximum, [1, 2, 1, 2, 1], 0.2)
        self.assertAlmostEqualOnAllArrayTypes(first_location_of_maximum, [1, 2, 1, 1, 2], 0.2)
        self.assertAlmostEqualOnAllArrayTypes(first_location_of_maximum, [2, 1, 1, 1, 1], 0.0)
        self.assertAlmostEqualOnAllArrayTypes(first_location_of_maximum, [1, 1, 1, 1, 1], 0.0)
        self.assertAlmostEqualOnAllArrayTypes(first_location_of_maximum, [1], 0.0)
        self.assertIsNanOnAllArrayTypes(first_location_of_maximum, [])

    def test_last_location_of_minimum(self):
        self.assertAlmostEqualOnAllArrayTypes(last_location_of_minimum, [1, 2, 1, 2, 1], 1.0)
        self.assertAlmostEqualOnAllArrayTypes(last_location_of_minimum, [1, 2, 1, 2, 2], 0.6)
        self.assertAlmostEqualOnAllArrayTypes(last_location_of_minimum, [2, 1, 1, 1, 2], 0.8)
        self.assertAlmostEqualOnAllArrayTypes(last_location_of_minimum, [1, 1, 1, 1, 1], 1.0)
        self.assertAlmostEqualOnAllArrayTypes(last_location_of_minimum, [1], 1.0)
        self.assertIsNanOnAllArrayTypes(last_location_of_minimum, [])

    def test_first_location_of_minimum(self):
        self.assertAlmostEqualOnAllArrayTypes(first_location_of_minimum, [1, 2, 1, 2, 1], 0.0)
        self.assertAlmostEqualOnAllArrayTypes(first_location_of_minimum, [2, 2, 1, 2, 2], 0.4)
        self.assertAlmostEqualOnAllArrayTypes(first_location_of_minimum, [2, 1, 1, 1, 2], 0.2)
        self.assertAlmostEqualOnAllArrayTypes(first_location_of_minimum, [1, 1, 1, 1, 1], 0.0)
        self.assertAlmostEqualOnAllArrayTypes(first_location_of_minimum, [1], 0.0)
        self.assertIsNanOnAllArrayTypes(first_location_of_minimum, [])

    def test_percentage_of_doubled_datapoints(self):
        self.assertAlmostEqualOnAllArrayTypes(percentage_of_reoccurring_datapoints_to_all_datapoints, [1, 1, 2, 3, 4], 0.25)
        self.assertAlmostEqualOnAllArrayTypes(percentage_of_reoccurring_datapoints_to_all_datapoints, [1, 1.5, 2, 3], 0)
        self.assertAlmostEqualOnAllArrayTypes(percentage_of_reoccurring_datapoints_to_all_datapoints, [1], 0)
        self.assertAlmostEqualOnAllArrayTypes(percentage_of_reoccurring_datapoints_to_all_datapoints, [1.111, -2.45, 1.111, 2.45], 1.0 / 3.0)
        self.assertIsNanOnAllArrayTypes(percentage_of_reoccurring_datapoints_to_all_datapoints, [])

    def test_ratio_of_doubled_values(self):
        self.assertAlmostEqualOnAllArrayTypes(percentage_of_reoccurring_values_to_all_values, [1, 1, 2, 3, 4], 0.4)
        self.assertAlmostEqualOnAllArrayTypes(percentage_of_reoccurring_values_to_all_values, [1, 1.5, 2, 3], 0)
        self.assertAlmostEqualOnAllArrayTypes(percentage_of_reoccurring_values_to_all_values, [1], 0)
        self.assertAlmostEqualOnAllArrayTypes(percentage_of_reoccurring_values_to_all_values, [1.111, -2.45, 1.111, 2.45], 0.5)
        self.assertIsNanOnAllArrayTypes(percentage_of_reoccurring_values_to_all_values, [])

    def test_sum_of_doubled_values(self):
        self.assertAlmostEqualOnAllArrayTypes(sum_of_reoccurring_values, [1, 1, 2, 3, 4], 2)
        self.assertAlmostEqualOnAllArrayTypes(sum_of_reoccurring_values, [1, 1.5, 2, 3], 0)
        self.assertAlmostEqualOnAllArrayTypes(sum_of_reoccurring_values, [1], 0)
        self.assertAlmostEqualOnAllArrayTypes(sum_of_reoccurring_values, [1.111, -2.45, 1.111, 2.45], 2.222)
        self.assertAlmostEqualOnAllArrayTypes(sum_of_reoccurring_values, [], 0)

    def test_uniqueness_factor(self):
        self.assertAlmostEqualOnAllArrayTypes(ratio_value_number_to_time_series_length, [1, 1, 2, 3, 4], 0.8)
        self.assertAlmostEqualOnAllArrayTypes(ratio_value_number_to_time_series_length, [1, 1.5, 2, 3], 1)
        self.assertAlmostEqualOnAllArrayTypes(ratio_value_number_to_time_series_length, [1], 1)
        self.assertAlmostEqualOnAllArrayTypes(ratio_value_number_to_time_series_length, [1.111, -2.45, 1.111, 2.45], 0.75)
        self.assertIsNanOnAllArrayTypes(ratio_value_number_to_time_series_length, [])

    def test_fft_coefficient(self):
        pass
        # todo: add unit test

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
        c = "TEST"
        param = [{"q": 0.5}]
        expected_index = ["TEST__index_mass_quantile__q_0.5"]
        res = index_mass_quantile(x, c, param)
        self.assertIsInstance(res, pd.Series)
        six.assertCountEqual(self, list(res.index), expected_index)
        self.assertAlmostEqual(res["TEST__index_mass_quantile__q_0.5"], 0.5, places=1)

        x = [0] * 1000 + [1]
        c = "TEST"
        param = [{"q": 0.5}, {"q": 0.99}]
        expected_index = ["TEST__index_mass_quantile__q_0.5", "TEST__index_mass_quantile__q_0.99"]
        res = index_mass_quantile(x, c, param)
        self.assertIsInstance(res, pd.Series)
        six.assertCountEqual(self, list(res.index), expected_index)
        self.assertAlmostEqual(res["TEST__index_mass_quantile__q_0.5"], 1, places=1)
        self.assertAlmostEqual(res["TEST__index_mass_quantile__q_0.99"], 1, places=1)

        x = [0, 1, 1, 0, 0, 1, 0, 0]
        c = "TEST"
        param = [{"q": 0.30}, {"q": 0.60}, {"q": 0.90}]
        expected_index = ["TEST__index_mass_quantile__q_0.3", "TEST__index_mass_quantile__q_0.6",
                          "TEST__index_mass_quantile__q_0.9"]
        res = index_mass_quantile(x, c, param)
        self.assertIsInstance(res, pd.Series)

        six.assertCountEqual(self, list(res.index), expected_index)
        self.assertAlmostEqual(res["TEST__index_mass_quantile__q_0.3"], 0.25, places=1)
        self.assertAlmostEqual(res["TEST__index_mass_quantile__q_0.6"], 0.375, places=1)
        self.assertAlmostEqual(res["TEST__index_mass_quantile__q_0.9"], 0.75, places=1)

        x = [0, 0, 0]
        c = "TEST"
        param = [{"q": 0.5}]
        expected_index = ["TEST__index_mass_quantile__q_0.5"]
        res = index_mass_quantile(x, c, param)
        self.assertIsInstance(res, pd.Series)
        six.assertCountEqual(self, list(res.index), expected_index)
        self.assertTrue(np.isnan(res["TEST__index_mass_quantile__q_0.5"]))

        x = []
        c = "TEST"
        param = [{"q": 0.5}]
        expected_index = ["TEST__index_mass_quantile__q_0.5"]
        res = index_mass_quantile(x, c, param)
        self.assertIsInstance(res, pd.Series)
        six.assertCountEqual(self, list(res.index), expected_index)
        self.assertTrue(np.isnan(res["TEST__index_mass_quantile__q_0.5"]))


    def test_number_cwt_peaks(self):
        pass
        # todo: add unit test

    def test_spkt_welch_density(self):
        pass
        # todo: add unit test

    def test_cwt_coefficients(self):
        x = [0.1, 0.2, 0.3]
        c = "TEST"
        param = [{"widths": (1, 2, 3), "coeff": 2, "w": 1},
                 {"widths": (1, 3), "coeff": 2, "w": 3},
                 {"widths": (1, 3), "coeff": 5, "w": 3}]
        shuffle(param)

        expected_index = ["TEST__cwt_coefficients__widths_(1, 2, 3)__coeff_2__w_1",
                          "TEST__cwt_coefficients__widths_(1, 3)__coeff_2__w_3",
                          "TEST__cwt_coefficients__widths_(1, 3)__coeff_5__w_3"]

        res = cwt_coefficients(x, c, param)

        # todo: add unit test for the values
        six.assertCountEqual(self, list(res.index), expected_index)
        self.assertTrue(math.isnan(res["TEST__cwt_coefficients__widths_(1, 3)__coeff_5__w_3"]))

    def test_ar_coefficient(self):

        # Test for X_i = 2.5 * X_{i-1} + 1
        c = "TEST"
        param = [{"k": 1, "coeff": 0}, {"k": 1, "coeff": 1}]
        shuffle(param)

        x = [1] + 9 * [0]
        for i in range(1, len(x)):
            x[i] = 2.5 * x[i - 1] + 1

        res = ar_coefficient(x, c, param)
        expected_index = ["TEST__ar_coefficient__k_1__coeff_0", "TEST__ar_coefficient__k_1__coeff_1"]

        self.assertIsInstance(res, pd.Series)
        six.assertCountEqual(self, list(res.index), expected_index)
        self.assertAlmostEqual(res["TEST__ar_coefficient__k_1__coeff_0"], 1, places=2)
        self.assertAlmostEqual(res["TEST__ar_coefficient__k_1__coeff_1"], 2.5, places=2)

        # Test for X_i = 1.4 * X_{i-1} - 1 X_{i-2} + 1
        c = "TEST"
        param = [{"k": 1, "coeff": 0}, {"k": 1, "coeff": 1},
                 {"k": 2, "coeff": 0}, {"k": 2, "coeff": 1}, {"k": 2, "coeff": 2}, {"k": 2, "coeff": 3}]
        shuffle(param)

        x = [1, 1] + 5 * [0]
        for i in range(2, len(x)):
            x[i] = (-2) * x[i-2] + 3.5 * x[i-1] + 1

        res = ar_coefficient(x, c, param)
        expected_index = ["TEST__ar_coefficient__k_1__coeff_0", "TEST__ar_coefficient__k_1__coeff_1",
                          "TEST__ar_coefficient__k_2__coeff_0", "TEST__ar_coefficient__k_2__coeff_1",
                          "TEST__ar_coefficient__k_2__coeff_2", "TEST__ar_coefficient__k_2__coeff_3"]

        print(res.sort_index())
        self.assertIsInstance(res, pd.Series)
        six.assertCountEqual(self, list(res.index), expected_index)
        self.assertAlmostEqual(res["TEST__ar_coefficient__k_2__coeff_0"], 1, places=2)
        self.assertAlmostEqual(res["TEST__ar_coefficient__k_2__coeff_1"], 3.5, places=2)
        self.assertAlmostEqual(res["TEST__ar_coefficient__k_2__coeff_2"], -2, places=2)
        self.assertTrue(np.isnan(res["TEST__ar_coefficient__k_2__coeff_3"]))

    def test_time_reversal_asymmetry_statistic(self):
        x = [1]*10
        self.assertAlmostEqualOnAllArrayTypes(time_reversal_asymmetry_statistic, x, 0, 0)
        self.assertAlmostEqualOnAllArrayTypes(time_reversal_asymmetry_statistic, x, 0, 1)
        self.assertAlmostEqualOnAllArrayTypes(time_reversal_asymmetry_statistic, x, 0, 2)
        self.assertAlmostEqualOnAllArrayTypes(time_reversal_asymmetry_statistic, x, 0, 3)

        x = [1, 2, -3, 4]
        # 1/2 * ( (4^2 * 2 + 3 * 2^2) + (3^2*1)-(2*1^1)) = 1/2 * (32+12+9-2) = 51/2
        self.assertAlmostEqualOnAllArrayTypes(time_reversal_asymmetry_statistic, x, 25.5, 1)
        # 4^2 * 1 - 2 * 1^2 = 16 -2
        self.assertAlmostEqualOnAllArrayTypes(time_reversal_asymmetry_statistic, x, 14, 2)
        self.assertAlmostEqualOnAllArrayTypes(time_reversal_asymmetry_statistic, x, 0, 3)

    def test_binned_entropy(self):
        self.assertAlmostEqualOnAllArrayTypes(binned_entropy, [10] * 100, 0, 10)
        self.assertAlmostEqualOnAllArrayTypes(binned_entropy, [10] * 10 + [1], - (10 / 11 * np.math.log(10 / 11) +
                                                                                  1 / 11 * np.math.log(1 / 11)), 10)
        self.assertAlmostEqualOnAllArrayTypes(binned_entropy, [10] * 10 + [1], - (10 / 11 * np.math.log(10 / 11) +
                                                                                  1 / 11 * np.math.log(1 / 11)), 10)
        self.assertAlmostEqualOnAllArrayTypes(binned_entropy, [10] * 10 + [1], - (10 / 11 * np.math.log(10 / 11) +
                                                                                  1 / 11 * np.math.log(1 / 11)), 100)
        self.assertAlmostEqualOnAllArrayTypes(binned_entropy, list(range(10)), - np.math.log(1 / 10), 100)
        self.assertAlmostEqualOnAllArrayTypes(binned_entropy, list(range(100)), - np.math.log(1 / 2), 2)

    
    def test_sample_entropy(self):
        ts = [1, 4, 5, 1, 7, 3, 1, 2, 5, 8, 9, 7, 3, 7, 9, 5, 4, 3, 9, 1, 2, 3, 4, 2, 9, 6, 7, 4, 9, 2, 9, 9, 6, 5, 1, 3, 8, 1, 5, 3, 8, 4, 1, 2, 2, 1, 6, 5, 3, 6, 5, 4, 8, 9, 6, 7, 5, 3, 2, 5, 4, 2, 5, 1, 6, 5, 3, 5, 6, 7, 8, 5, 2, 8, 6, 3, 8, 2, 7, 1, 7, 3, 5, 6, 2, 1, 3, 7, 3, 5, 3, 7, 6, 7, 7, 2, 3, 1, 7, 8]
        self.assertAlmostEqualOnAllArrayTypes(sample_entropy, ts, 2.21187685)
    

    def test_autocorrelation(self):
        self.assertAlmostEqualOnAllArrayTypes(autocorrelation, [1, 2, 1, 2, 1, 2], -1, 1)
        self.assertAlmostEqualOnAllArrayTypes(autocorrelation, [1, 2, 1, 2, 1, 2], 1, 2)
        self.assertAlmostEqualOnAllArrayTypes(autocorrelation, [1, 2, 1, 2, 1, 2], -1, 3)
        self.assertAlmostEqualOnAllArrayTypes(autocorrelation, [1, 2, 1, 2, 1, 2], 1, 4)
        self.assertIsNanOnAllArrayTypes(autocorrelation, [1, 2, 1, 2, 1, 2], 200)

    def test_quantile(self):
        self.assertAlmostEqualOnAllArrayTypes(quantile, [1, 1, 1, 3, 4, 7, 9, 11, 13, 13], 1.0, 0.2)
        self.assertAlmostEqualOnAllArrayTypes(quantile, [1, 1, 1, 3, 4, 7, 9, 11, 13, 13], 13, 0.9)
        self.assertAlmostEqualOnAllArrayTypes(quantile, [1, 1, 1, 3, 4, 7, 9, 11, 13, 13], 13, 1.0)
        self.assertAlmostEqualOnAllArrayTypes(quantile, [1], 1, 0.5)
        self.assertIsNanOnAllArrayTypes(quantile, [], 0.5)

    def test_mean_abs_change_quantiles(self):

        self.assertAlmostEqualOnAllArrayTypes(mean_abs_change_quantiles, list(range(10)), 1, ql=0.1, qh=0.9)
        self.assertAlmostEqualOnAllArrayTypes(mean_abs_change_quantiles, list(range(10)), 0, ql=0.15, qh=0.18)
        self.assertAlmostEqualOnAllArrayTypes(mean_abs_change_quantiles, [0, 1, 0, 0, 0], 0.5, ql=0, qh=1)
        self.assertAlmostEqualOnAllArrayTypes(mean_abs_change_quantiles, [0, 1, 0, 0, 0], 0.5, ql=0.1, qh=1)
        self.assertAlmostEqualOnAllArrayTypes(mean_abs_change_quantiles, [0, 1, 0, 0, 0], 0, ql=0.1, qh=0.6)
        self.assertAlmostEqualOnAllArrayTypes(mean_abs_change_quantiles, [0, 1, -9, 0, 0], 5, ql=0, qh=1)
        self.assertAlmostEqualOnAllArrayTypes(mean_abs_change_quantiles, [0, 1, -9, 0, 0], 0.5, ql=0.1, qh=1)
        self.assertAlmostEqualOnAllArrayTypes(mean_abs_change_quantiles, [0, 1, -9, 0, 0, 1, 0], 0.75, ql=0.1, qh=1)

    def test_value_count(self):
        self.assertEqualPandasSeriesWrapper(value_count, [1] * 10, 10, value=1)
        self.assertEqualPandasSeriesWrapper(value_count, list(range(10)), 1, value=0)
        self.assertEqualPandasSeriesWrapper(value_count, [1] * 10, 0, value=0)
        self.assertEqualPandasSeriesWrapper(value_count, [np.NaN, 0, 1] * 3, 3, value=0)
        self.assertEqualPandasSeriesWrapper(value_count, [np.NINF, 0, 1] * 3, 3, value=0)
        self.assertEqualPandasSeriesWrapper(value_count, [np.PINF, 0, 1] * 3, 3, value=0)
        self.assertEqualPandasSeriesWrapper(value_count, [0.1, 0.2, 0.3] * 3, 3, value=0.2)
        self.assertEqualPandasSeriesWrapper(value_count, [np.NaN, 0, 1] * 3, 3, value=np.NaN)
        self.assertEqualPandasSeriesWrapper(value_count, [np.NINF, 0, 1] * 3, 3, value=np.NINF)
        self.assertEqualPandasSeriesWrapper(value_count, [np.PINF, 0, 1] * 3, 3, value=np.PINF)

    def test_range_count(self):
        self.assertEqualPandasSeriesWrapper(range_count, [1] * 10, 0, min=1, max=1)
        self.assertEqualPandasSeriesWrapper(range_count, [1] * 10, 0, min=0.9, max=1)
        self.assertEqualPandasSeriesWrapper(range_count, [1] * 10, 10, min=1, max=1.1)
        self.assertEqualPandasSeriesWrapper(range_count, list(range(10)), 9, min=0, max=9)
        self.assertEqualPandasSeriesWrapper(range_count, list(range(10)), 10, min=0, max=10)
        self.assertEqualPandasSeriesWrapper(range_count, list(range(0, -10, -1)), 9, min=-10, max=0)
        self.assertEqualPandasSeriesWrapper(range_count, [np.NaN, np.PINF, np.NINF] + list(range(10)), 10, min=0, max=10)

    def test_approximate_entropy(self):
        self.assertEqualOnAllArrayTypes(approximate_entropy, [1], 0, m=2, r=0.5)
        self.assertEqualOnAllArrayTypes(approximate_entropy, [1, 2], 0, m=2, r=0.5)
        self.assertEqualOnAllArrayTypes(approximate_entropy, [1, 2, 3], 0, m=2, r=0.5)
        self.assertEqualOnAllArrayTypes(approximate_entropy, [1, 2, 3], 0, m=2, r=0.5)
        self.assertAlmostEqualOnAllArrayTypes(approximate_entropy, [12, 13, 15, 16, 17]*10, 0.282456191, m=2, r=0.9)
        self.assertRaises(ValueError, approximate_entropy, x=[12, 13, 15, 16, 17]*10, m=2, r=-0.5)
