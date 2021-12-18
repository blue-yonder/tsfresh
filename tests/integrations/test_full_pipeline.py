# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import os
import shutil
import tempfile
from unittest import TestCase

import pandas as pd
from sklearn.pipeline import Pipeline

from tests.fixtures import warning_free
from tsfresh.examples.robot_execution_failures import (
    download_robot_execution_failures,
    load_robot_execution_failures,
)
from tsfresh.transformers import RelevantFeatureAugmenter


class FullPipelineTestCase_robot_failures(TestCase):
    def setUp(self):
        self.temporary_folder = tempfile.mkdtemp()
        temporary_file = os.path.join(self.temporary_folder, "data")

        download_robot_execution_failures(file_name=temporary_file)
        self.timeseries, self.y = load_robot_execution_failures(
            file_name=temporary_file
        )
        self.df = pd.DataFrame(index=self.timeseries.id.unique())

        # shrink the time series for this test
        self.timeseries = self.timeseries[["id", "time", "F_x"]]

    def tearDown(self):
        shutil.rmtree(self.temporary_folder)

    def test_relevant_extraction(self):
        self.assertGreater(len(self.y), 0)
        self.assertGreater(len(self.df), 0)
        self.assertGreater(len(self.timeseries), 0)

        relevant_augmenter = RelevantFeatureAugmenter(
            column_id="id", column_sort="time"
        )
        relevant_augmenter.set_timeseries_container(self.timeseries)

        pipe = Pipeline([("relevant_augmenter", relevant_augmenter)])

        with warning_free():
            pipe.fit(self.df, self.y)
        extracted_features = pipe.transform(self.df)

        some_expected_features = {
            "F_x__abs_energy",
            "F_x__absolute_sum_of_changes",
            "F_x__autocorrelation__lag_1",
            "F_x__binned_entropy__max_bins_10",
            "F_x__count_above_mean",
            "F_x__longest_strike_above_mean",
            "F_x__maximum",
            "F_x__mean_abs_change",
            "F_x__minimum",
            "F_x__quantile__q_0.1",
            "F_x__range_count__max_1__min_-1",
            "F_x__spkt_welch_density__coeff_2",
            "F_x__standard_deviation",
            "F_x__value_count__value_0",
            "F_x__variance",
            "F_x__variance_larger_than_standard_deviation",
        }

        self.assertGreaterEqual(set(extracted_features.columns), some_expected_features)
        self.assertGreater(len(extracted_features), 0)
