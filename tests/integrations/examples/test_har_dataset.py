# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import shutil
import tempfile
from unittest import TestCase

from pandas import DataFrame, Series

from tsfresh.examples.har_dataset import (
    download_har_dataset,
    load_har_classes,
    load_har_dataset,
)


class HumanActivityTestCase(TestCase):
    def setUp(self):
        self.temporary_folder = tempfile.mkdtemp()

        download_har_dataset(folder_name=self.temporary_folder)
        self.data = load_har_dataset(folder_name=self.temporary_folder)
        self.classes = load_har_classes(folder_name=self.temporary_folder)

    def tearDown(self):
        shutil.rmtree(self.temporary_folder)

    def test_characteristics_downloaded_robot_execution_failures(self):
        self.assertEqual(len(self.data), 7352)
        self.assertIsInstance(self.data, DataFrame)

        self.assertEqual(len(self.classes), 7352)
        self.assertIsInstance(self.classes, Series)

    def test_index(self):
        self.assertCountEqual(self.data.index, self.classes.index)
