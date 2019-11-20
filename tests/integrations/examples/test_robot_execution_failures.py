# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from unittest import TestCase

from tsfresh import extract_features
from tsfresh.examples.robot_execution_failures import load_robot_execution_failures, download_robot_execution_failures
from pandas import DataFrame, Series
import numpy as np


class RobotExecutionFailuresTestCase(TestCase):
    def setUp(self):
        download_robot_execution_failures()
        self.X, self.y = load_robot_execution_failures()

    def test_characteristics_downloaded_robot_execution_failures(self):
        self.assertEqual(len(self.X), 1320)
        self.assertIsInstance(self.X, DataFrame)
        self.assertIsInstance(self.y, Series)
        self.assertCountEqual(['id', 'time', 'F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z'], list(self.X.columns))

    def test_extraction_runs_through(self):
        df = extract_features(self.X[self.X.id < 3], column_id="id", column_sort="time")

        self.assertCountEqual(df.index.values, [1, 2])
        self.assertGreater(len(df), 0)

    def test_binary_target_is_default(self):
        _, y = load_robot_execution_failures()

        assert len(y.unique()) == 2

    def test_multilabel_target_on_request(self):
        _, y = load_robot_execution_failures(multiclass=True)

        assert len(y.unique()) > 2
        assert y.dtype == np.object
