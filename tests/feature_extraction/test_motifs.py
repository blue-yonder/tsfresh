# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# This module Ezekiel Kruglick 2017

from __future__ import absolute_import, division
from __future__ import print_function
import unittest
import numpy as np
from tsfresh.feature_extraction import motifs
import six

sample_array1 = np.array([9., -63., 5., 157., -21., -20., 27., -72., -123.,
                          94., 154., -57., -48., 18., -5., 14., 5., -7.,
                          -5., 13., -3., -1., 6., 4., -8., -4., 10.,
                          8., -2., 1., 2., -9.])

sample_array2 = np.array([-20., -111., 69., 238., -30., -56., -5., -31., -95.,
                          60., 148., -52., -52., 14., -11., 18., 7., -8.,
                          -5., 11., -1., 1., 1., 5., 0., -9., -2.,
                          17., 1., -2., 3., -14.])


class TestMotifSubelements(unittest.TestCase):
    def setUp(self):
        self.length = 3

    def test_distance_calculator_for_identical_arrays(self):
        test_array1 = np.ones(50, dtype=np.float)
        test_array2 = np.ones(50, dtype=np.float)
        answer = motifs.distance(test_array1, test_array2)
        self.assertEqual(answer, 0.0)

    def test_distance_calculator_for_simple_different_arrays(self):
        test_array1 = np.ones(50, dtype=np.float)
        test_array2 = 2 * np.ones(50, dtype=np.float)
        answer = motifs.distance(test_array1, test_array2)
        self.assertAlmostEqual(answer, np.sqrt(50))

    def test_distance_calculator_for_complex_different_arrays(self):
        answer = motifs.distance(sample_array1, sample_array2)
        self.assertAlmostEqual(answer, np.sqrt(20370))

    def test_sliding_window(self):
        answer = motifs._sliding_window(sample_array1, 5)
        self.assertItemsEqual(answer[1], [-63., 5., 157., -21., -20.])
        self.assertItemsEqual(answer[-1], [8., -2., 1., 2., -9.])

    def test_match_scores(self):
        answer = motifs._match_scores(sample_array1, sample_array1[5:10])
        self.assertEqual(answer[5], 0.0)

    def test_candidate_duplicate_removal(self):
        candidates = [(1862, 1984, 4.4220442590667863),
                      (1984, 1862, 4.4220442590667863),
                      (1863, 1985, 4.4960457316429485),
                      (1985, 1863, 4.4960457316429485),
                      (73, 1762, 4.8654597310856209),
                      (1762, 73, 4.8654597310856209),
                      (65, 62, 5.093744473040184),
                      (1904, 1898, 5.1468554369121344),
                      (64, 61, 5.1830888985252646),
                      (1888, 1885, 5.2769356323613037)]
        answer = motifs._candidates_top_uniques(self.length, candidates, 4)
        self.assertNotEqual(answer[0][1], answer[1][0])
        self.assertNotEqual(answer[0][1], answer[2][0])
        self.assertNotEqual(answer[0][1], answer[3][0])

    def test_find_motifs_a(self):
        found_motifs = motifs.find_motifs(self.length, sample_array1, 5)
        self.assertEqual(len(found_motifs), 5)

    def test_find_motifs_b(self):
        self.length = 10
        series = np.concatenate([sample_array1, sample_array1, sample_array1])
        found_motifs = motifs.find_motifs(self.length, series, 5)
        self.assertEqual(len(found_motifs), 4)
        # The pattern below is generated because the series is made of repeating sample arrays so they will always
        # match on the period of the sample array
        self.assertTrue(all([x[1] - x[0] == 32 for x in found_motifs]))

    def test_count_motifs(self):
        found_motifs = [(16, 23, 1.7320508075688772), (17, 24, 3.3166247903553998), (20, 28, 4.5825756949558398),
                        (14, 25, 5.0990195135927845), (19, 27, 5.4772255750516612)]
        series = np.concatenate([sample_array1, sample_array1, sample_array1])
        count = motifs.count_motifs(series, found_motifs[0], dist=15)
        self.assertIsInstance(count, six.integer_types)
        self.assertEqual(count, 25)


if __name__ == '__main__':
    unittest.main()
