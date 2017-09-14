# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from unittest import TestCase
import numpy as np
from tsfresh.utilities.distribution import Distributor


class DistributorTestCase(TestCase):

    def test_partion(self):

        distributor = Distributor()
        data = [1, 3, 10, -10, 343.0]
        distro = distributor.partition(data, 3)

        self.assertEqual(distro.next(), [1, 3, 10])
        self.assertEqual(distro.next(), [-10, 343.0])

        data = np.arange(10)
        distro = distributor.partition(data, 2)

        self.assertEqual(distro.next(), [0, 1])
        self.assertEqual(distro.next(), [2, 3])