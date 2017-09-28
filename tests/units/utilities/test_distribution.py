# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from unittest import TestCase
import numpy as np
from tsfresh.utilities.distribution import MapDistributor


class MapDistributorTestCase(TestCase):

    def test_partion(self):

        distributor = MapDistributor()

        data = [1, 3, 10, -10, 343.0]
        distro = distributor.partition(data, 3)
        self.assertEqual(next(distro), [1, 3, 10])
        self.assertEqual(next(distro), [-10, 343.0])

        data = np.arange(10)
        distro = distributor.partition(data, 2)
        self.assertEqual(next(distro), [0, 1])
        self.assertEqual(next(distro), [2, 3])

    def test__calculate_best_chunksize(self):

        distributor = MapDistributor(n_workers=2)
        self.assertEqual(distributor._calculate_best_chunksize(10), 1)
        self.assertEqual(distributor._calculate_best_chunksize(11), 2)
        self.assertEqual(distributor._calculate_best_chunksize(100), 10)
        self.assertEqual(distributor._calculate_best_chunksize(101), 11)

        distributor = MapDistributor(n_workers=3)
        self.assertEqual(distributor._calculate_best_chunksize(10), 1)
        self.assertEqual(distributor._calculate_best_chunksize(30), 2)
        self.assertEqual(distributor._calculate_best_chunksize(31), 3)
