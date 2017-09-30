# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from unittest import TestCase
import numpy as np
from tsfresh.utilities.distribution import MultiprocessingDistributor


class MultiprocessingDistributorTestCase(TestCase):

    def test_partion(self):

        distributor = MultiprocessingDistributor(n_workers=1)

        data = [1, 3, 10, -10, 343.0]
        distro = distributor.partition(data, 3)
        self.assertEqual(next(distro), [1, 3, 10])
        self.assertEqual(next(distro), [-10, 343.0])

        data = np.arange(10)
        distro = distributor.partition(data, 2)
        self.assertEqual(next(distro), [0, 1])
        self.assertEqual(next(distro), [2, 3])

    def test__calculate_best_chunk_size(self):

        distributor = MultiprocessingDistributor(n_workers=2)
        self.assertEqual(distributor.calculate_best_chunk_size(10), 1)
        self.assertEqual(distributor.calculate_best_chunk_size(11), 2)
        self.assertEqual(distributor.calculate_best_chunk_size(100), 10)
        self.assertEqual(distributor.calculate_best_chunk_size(101), 11)

        distributor = MultiprocessingDistributor(n_workers=3)
        self.assertEqual(distributor.calculate_best_chunk_size(10), 1)
        self.assertEqual(distributor.calculate_best_chunk_size(30), 2)
        self.assertEqual(distributor.calculate_best_chunk_size(31), 3)


    # todo: test distribute
    # todo: test map
    # todo: test close
    # todo: test dask


    # todo: error for test not valid distributor
    # todo: test ipaddresses at clusterdaskdistributor