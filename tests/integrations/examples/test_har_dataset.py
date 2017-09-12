# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from unittest import TestCase
from tsfresh.examples.har_dataset import download_har_dataset, load_har_dataset, load_har_classes
from pandas import DataFrame, Series
import six

class HumanActivityTestCase(TestCase):
    def setUp(self):
        download_har_dataset()
        self.data = load_har_dataset()
        self.classes = load_har_classes()

    def test_characteristics_downloaded_robot_execution_failures(self):
        self.assertEqual(len(self.data), 7352)
        self.assertIsInstance(self.data, DataFrame)

        self.assertEqual(len(self.classes), 7352)
        self.assertIsInstance(self.classes, Series)

    def test_index(self):
        six.assertCountEqual(self, self.data.index, self.classes.index)