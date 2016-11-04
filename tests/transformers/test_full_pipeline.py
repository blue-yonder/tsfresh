# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from unittest import TestCase
import pandas as pd
from sklearn.pipeline import Pipeline
from tsfresh.examples.robot_execution_failures import load_robot_execution_failures, download_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter


class FullPipelineTestCase_robot_failures(TestCase):
    def setUp(self):
        download_robot_execution_failures()
        self.timeseries, self.y = load_robot_execution_failures()
        self.df = pd.DataFrame(index=self.timeseries.id.unique())

        # shrink the time series for this test
        self.timeseries = self.timeseries[["id", "time", "a"]]

    def test_relevant_extraction(self):
        self.assertGreater(len(self.y), 0)
        self.assertGreater(len(self.df), 0)
        self.assertGreater(len(self.timeseries), 0)

        relevant_augmenter = RelevantFeatureAugmenter(column_id="id", column_sort="time")
        relevant_augmenter.set_timeseries_container(self.timeseries)

        pipe = Pipeline([("relevant_augmenter", relevant_augmenter)])

        pipe.fit(self.df, self.y)
        extracted_features = pipe.transform(self.df)

        some_expected_features = {'a__abs_energy',
                                  'a__absolute_sum_of_changes',
                                  'a__ar_coefficient__k_10__coeff_0',
                                  'a__autocorrelation__lag_1',
                                  'a__binned_entropy__max_bins_10',
                                  'a__count_above_mean',
                                  'a__index_mass_quantile__q_0.1',
                                  'a__index_mass_quantile__q_0.2',
                                  'a__index_mass_quantile__q_0.3',
                                  'a__index_mass_quantile__q_0.4',
                                  'a__index_mass_quantile__q_0.6',
                                  'a__index_mass_quantile__q_0.7',
                                  'a__index_mass_quantile__q_0.8',
                                  'a__index_mass_quantile__q_0.9',
                                  'a__longest_strike_above_mean',
                                  'a__maximum',
                                  'a__mean_abs_change',
                                  'a__mean_abs_change_quantiles__qh_0.2__ql_0.0',
                                  'a__mean_abs_change_quantiles__qh_0.4__ql_0.0',
                                  'a__mean_abs_change_quantiles__qh_0.4__ql_0.2',
                                  'a__mean_abs_change_quantiles__qh_0.6__ql_0.0',
                                  'a__mean_abs_change_quantiles__qh_0.6__ql_0.2',
                                  'a__mean_abs_change_quantiles__qh_0.6__ql_0.4',
                                  'a__mean_abs_change_quantiles__qh_0.8__ql_0.0',
                                  'a__mean_abs_change_quantiles__qh_0.8__ql_0.2',
                                  'a__mean_abs_change_quantiles__qh_0.8__ql_0.4',
                                  'a__mean_abs_change_quantiles__qh_1.0__ql_0.0',
                                  'a__mean_abs_change_quantiles__qh_1.0__ql_0.2',
                                  'a__mean_abs_change_quantiles__qh_1.0__ql_0.4',
                                  'a__mean_abs_change_quantiles__qh_1.0__ql_0.6',
                                  'a__mean_abs_change_quantiles__qh_1.0__ql_0.8',
                                  'a__mean_autocorrelation',
                                  'a__minimum',
                                  'a__quantile__q_0.1',
                                  'a__range_count__max_1__min_-1',
                                  'a__spkt_welch_density__coeff_2',
                                  'a__standard_deviation',
                                  'a__value_count__value_0',
                                  'a__variance',
                                  'a__variance_larger_than_standard_deviation'}

        self.assertGreaterEqual(set(extracted_features.columns), some_expected_features)
        self.assertGreater(len(extracted_features), 0)