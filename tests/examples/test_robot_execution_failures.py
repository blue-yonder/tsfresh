# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from unittest import TestCase
from tsfresh.examples.robot_execution_failures import load_robot_execution_failures, download_robot_execution_failures
from pandas import DataFrame, Series
import six

class RobotExecutionFailuresTestCase(TestCase):
    def setUp(self):
        download_robot_execution_failures()
        self.X, self.y = load_robot_execution_failures()


    def test_characteristics_downloaded_robot_execution_failures(self):
        self.assertEqual(len(self.X), 1320)
        self.assertIsInstance(self.X, DataFrame)
        self.assertIsInstance(self.y, Series)
        six.assertCountEqual(self, ['id', 'time', 'a', 'b', 'c', 'd', 'e', 'f'], list(self.X.columns))