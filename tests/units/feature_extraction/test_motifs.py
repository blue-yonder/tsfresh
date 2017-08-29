# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# This module Ezekiel Kruglick 2017

from __future__ import absolute_import, division, print_function

import unittest
import six
import numpy as np
import numpy.testing as npt
from tsfresh.feature_extraction import motifs
import pandas as pd


class TestMotifSubelements(unittest.TestCase):
    def setUp(self):
        self.length = 3
        self.s1 = np.array([9., -63., 5., 157., -21., -20., 27., -72., -123.,
                            94., 154., -57., -48., 18., -5., 14., 5., -7.,
                            -5., 13., -3., -1., 6., 4., -8., -4., 10.,
                            8., -2., 1., 2., -9.])

        self.s2 = np.array([-20., -111., 69., 238., -30., -56., -5., -31., -95.,
                            60., 148., -52., -52., 14., -11., 18., 7., -8.,
                            -5., 11., -1., 1., 1., 5., 0., -9., -2.,
                            17., 1., -2., 3., -14.])

    def test_distance_calculator_for_identical_arrays(self):
        test_array1 = np.ones(50, dtype=np.float)
        test_array2 = np.ones(50, dtype=np.float)
        answer = motifs.distance(test_array1, test_array2)
        self.assertEqual(answer, 0.0)

        test_array1 = np.random.normal(size=10)
        test_array2 = np.copy(test_array1)
        answer = motifs.distance(test_array1, test_array2)
        self.assertEqual(answer, 0.0)

    def test_distance_calculator_for_simple_different_arrays(self):
        test_array1 = np.ones(50, dtype=np.float)
        test_array2 = 2 * np.ones(50, dtype=np.float)
        answer = motifs.distance(test_array1, test_array2, type="euclid")
        self.assertAlmostEqual(answer, np.sqrt(50))

    def test_distance_calculator_for_complex_different_arrays(self):
        answer = motifs.distance(self.s1, self.s2)
        self.assertAlmostEqual(answer, np.sqrt(20370))

    def test_sliding_window(self):
        answer = motifs._array_of_sliding_windows(self.s1, 5)
        six.assertCountEqual(self, answer[1], [-63., 5., 157., -21., -20.])
        six.assertCountEqual(self, answer[-1], [8., -2., 1., 2., -9.])

        data = np.arange(5)
        expected_result = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        npt.assert_array_equal(expected_result, motifs._array_of_sliding_windows(data, pattern_length=2))

        data = pd.Series(np.arange(5))
        expected_result = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        npt.assert_array_equal(expected_result, motifs._array_of_sliding_windows(data, pattern_length=2))

    def test_match_scores(self):
        answer = motifs._match_scores(self.s1, self.s1[5:10])
        self.assertEqual(answer[5], 0.0)

        data = np.array([0, 1, 2])
        pattern = [0, 1, 2]
        result = motifs._match_scores(data, pattern)
        expected_result = [0]
        self.assertEqual(result, expected_result)

        data = np.array([0, 1, 2])
        pattern = [0, 1, -1]
        result = motifs._match_scores(data, pattern)
        expected_result = [3]
        self.assertEqual(result, expected_result)

        data = np.array([-2, 1, -3])
        pattern = [-2, 1]
        result = motifs._match_scores(data, pattern)
        expected_result = [0, 5]
        self.assertListEqual(list(result), expected_result)

    def test_generate_candidates(self):
        answer = motifs._generate_candidates(self.s1, self.length)
        self.assertEqual(len(answer),23)

    def test_candidate_duplicate_removal(self):
        candidates = [(1962, 1984, 4.4220442590667863),
                      (1964, 1986, 4.4220442590667863),
                      (1863, 1885, 4.4960457316429485),
                      (73, 95, 4.8654597310856209),
                      (72, 94, 4.654597310856209),
                      (62, 84, 5.093744473040184),
                      (1904, 1926, 5.1468554369121344),
                      (1888, 1910, 5.2769356323613037)]
        answer = motifs._candidates_top_uniques(candidates[0][1]-candidates[0][0], candidates, 4)
        # testing with set comparisons is nice and general way to check for overlaps
        set_1 = set(list(range(answer[0][0], answer[0][1])))
        set_2 = set(list(range(answer[1][0], answer[1][1])))
        set_3 = set(list(range(answer[2][0], answer[2][1])))
        set_4 = set(list(range(answer[3][0], answer[3][1])))
        self.assertFalse(any([
            set_1 & set_2,
            set_1 & set_3,
            set_1 & set_4,
            set_2 & set_3,
            set_2 & set_4,
            set_3 & set_4
        ]))

    def test_find_motifs_a(self):
        found_motifs = motifs.find_motifs(self.s1, self.length, 5)
        self.assertEqual(len(found_motifs), 5)

    def test_find_motifs_b(self):
        series = np.concatenate([self.s1, self.s1, self.s1])
        found_motifs = motifs.find_motifs(series, motif_length=8, motif_count=3)
        self.assertEqual(len(found_motifs), 3)
        # The pattern below is generated because the series is made of repeating sample arrays so they will always
        # match on the period of the sample array
        self.assertTrue(all([x[1] - x[0] == 32 for x in found_motifs]))

    def test_count_motifs(self):
        found_motifs = [(16, 23, 1.7320508075688772), (17, 24, 3.3166247903553998), (20, 28, 4.5825756949558398),
                        (14, 25, 5.0990195135927845), (19, 27, 5.4772255750516612)]
        series = np.concatenate([self.s1, self.s1, self.s1])
        count = motifs.count_motifs(series, found_motifs[0], dist=15)
        self.assertIsInstance(count, six.integer_types)
        self.assertEqual(count, 25)
