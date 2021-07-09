# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import warnings

import mock
import numpy as np
import pandas as pd
import pytest

from tsfresh.feature_selection.relevance import (
    calculate_relevance_table,
    combine_relevance_tables,
    get_feature_type,
    infer_ml_task,
)


class TestInferMLTask:
    def test_infers_classification_for_integer_target(self):
        y = pd.Series([1, 2, 3])
        assert "classification" == infer_ml_task(y)

    def test_infers_classification_for_boolean_target(self):
        y = pd.Series([True, False, False])
        assert "classification" == infer_ml_task(y)

    def test_infers_classification_for_object_target(self):
        y = pd.Series(["high", "low"])
        assert "classification" == infer_ml_task(y)

    def test_infers_regression_for_float_target(self):
        y = pd.Series([1.0, 1.5, 1.7])
        assert "regression" == infer_ml_task(y)


class TestCalculateRelevanceTable:
    @pytest.fixture()
    def y_binary(self):
        return pd.Series([0, 1, 1])

    @pytest.fixture()
    def y_real(self):
        return pd.Series([0.1, 0.2, 0.1])

    @pytest.fixture()
    def y_multi(self):
        y0 = np.zeros(100)
        y1 = np.repeat(1, 100)
        y2 = np.repeat(2, 100)
        return pd.Series(np.uint8(np.concatenate([y0, y1, y2])))

    @pytest.fixture()
    def X(self):
        df = pd.DataFrame()
        df["feature_binary"] = [1, 1, 0]
        df["feature_real"] = [0.1, 0.2, 0.3]
        return df

    def test_restrict_ml_task_options(self, X, y_binary):
        with pytest.raises(ValueError):
            calculate_relevance_table(X, y_binary, ml_task="some_other_task")

    def test_constant_feature_irrelevant(self, y_binary):
        X = pd.DataFrame([1, 1, 1], columns=["feature_binary"])

        relevance_table = calculate_relevance_table(X, y_binary)
        assert "feature_binary" == relevance_table.index[0]
        assert "constant" == relevance_table.type[0]
        assert np.isnan(relevance_table.p_value[0])
        assert not relevance_table.relevant[0]

    @mock.patch("tsfresh.feature_selection.relevance.target_binary_feature_real_test")
    @mock.patch("tsfresh.feature_selection.relevance.target_binary_feature_binary_test")
    def test_target_binary_calls_correct_tests(
        self,
        significance_test_feature_binary_mock,
        significance_test_feature_real_mock,
        X,
        y_binary,
    ):
        significance_test_feature_binary_mock.return_value = 0.5
        significance_test_feature_real_mock.return_value = 0.7
        relevance_table = calculate_relevance_table(X, y_binary, n_jobs=0)

        assert 0.5 == relevance_table.loc["feature_binary"].p_value
        assert 0.7 == relevance_table.loc["feature_real"].p_value
        assert 2 == significance_test_feature_binary_mock.call_count
        assert 2 == significance_test_feature_real_mock.call_count

    @mock.patch("tsfresh.feature_selection.relevance.target_real_feature_real_test")
    @mock.patch("tsfresh.feature_selection.relevance.target_real_feature_binary_test")
    def test_target_real_calls_correct_tests(
        self,
        significance_test_feature_binary_mock,
        significance_test_feature_real_mock,
        X,
        y_real,
    ):
        significance_test_feature_binary_mock.return_value = 0.5
        significance_test_feature_real_mock.return_value = 0.7

        relevance_table = calculate_relevance_table(X, y_real, n_jobs=0)

        assert 0.5 == relevance_table.loc["feature_binary"].p_value
        assert 0.7 == relevance_table.loc["feature_real"].p_value

        assert significance_test_feature_binary_mock.call_count == 1
        pd.testing.assert_series_equal(
            significance_test_feature_binary_mock.call_args[0][0], X["feature_binary"]
        )
        pd.testing.assert_series_equal(
            significance_test_feature_binary_mock.call_args[1]["y"], y_real
        )

        assert significance_test_feature_real_mock.call_count == 1
        pd.testing.assert_series_equal(
            significance_test_feature_real_mock.call_args[0][0], X["feature_real"]
        )
        pd.testing.assert_series_equal(
            significance_test_feature_real_mock.call_args[1]["y"], y_real
        )

    @mock.patch("tsfresh.feature_selection.relevance.target_real_feature_real_test")
    @mock.patch("tsfresh.feature_selection.relevance.target_real_feature_binary_test")
    def test_warning_for_no_relevant_feature(
        self,
        significance_test_feature_binary_mock,
        significance_test_feature_real_mock,
        X,
        y_real,
    ):
        significance_test_feature_binary_mock.return_value = 0.95
        significance_test_feature_real_mock.return_value = 0.95

        with pytest.warns(RuntimeWarning) as record:
            _ = calculate_relevance_table(
                X, y_real, n_jobs=0, ml_task="regression", show_warnings=True
            )
            assert len(record) >= 1
            assert (
                "No feature was found relevant for regression for fdr level = 0.05 (which corresponds "
                "to the maximal percentage of irrelevant features, consider using an higher fdr level "
                "or add other features."
            ) in [str(warning.message) for warning in record]

    def test_multiclass_requires_classification(self, X, y_real):
        with pytest.raises(AssertionError):
            calculate_relevance_table(X, y_real, multiclass=True, ml_task="regression")

    def test_multiclass_n_significant_error(self, X, y_binary):
        with pytest.raises(AssertionError):
            calculate_relevance_table(
                X, y_binary, multiclass=True, n_significant=3, ml_task="classification"
            )

    def test_multiclass_relevance_table_columns(self, X, y_binary):
        y = y_binary.copy()
        y[2] = 2
        relevance_table = calculate_relevance_table(X, y, multiclass=True)

        assert len(relevance_table.columns) == 10

    def test_multiclass_correct_features_relevant(self, y_multi):
        X_multi = pd.DataFrame()
        X_multi["relevant_0"] = np.concatenate([np.zeros(298), np.array([0.01, -0.01])])
        X_multi["relevant_3"] = X_multi["relevant_0"].copy()
        X_multi["relevant_3"][y_multi == 0] = np.random.uniform(2, 3, 100)
        X_multi["relevant_2"] = X_multi["relevant_3"].copy()
        X_multi["relevant_2"][y_multi == 1] = np.random.uniform(-2, -1, 100)

        relevance_table = calculate_relevance_table(
            X_multi, y_multi, multiclass=True, ml_task="classification", n_significant=3
        )
        assert relevance_table.loc["relevant_3", "relevant"]
        assert not relevance_table.loc["relevant_2", "relevant"]
        assert not relevance_table.loc["relevant_0", "relevant"]

        # the distributions of all 3 classes under a one vs. rest scheme will be separated enough for
        # this feature to be relevant for predicting 3 classes
        assert relevance_table.loc["relevant_3", "n_significant"] == 3

        # due to the distribution of this feature where y_multi == 0 being contained inside the range of
        # y_multi != 0 it will not pass the Mann-Whitney U test under a one vs. rest scheme for that class
        assert relevance_table.loc["relevant_2", "n_significant"] == 2
        assert relevance_table.loc["relevant_0", "n_significant"] == 0


class TestCombineRelevanceTables:
    @pytest.fixture()
    def relevance_table(self):
        relevance_table = pd.DataFrame(
            index=pd.Series(["f1", "f2", "f3", "f4"], name="feature")
        )
        relevance_table["relevant"] = [True, False, True, False]
        relevance_table["type"] = ["real"] * 4
        relevance_table["p_value"] = [0.1, 0.2, 0.3, 0.4]
        return relevance_table

    def test_disjuncts_relevance(self, relevance_table):
        relevance_table_2 = relevance_table.copy()
        relevance_table_2.relevant = [False, True, True, False]
        result = combine_relevance_tables([relevance_table, relevance_table_2])

        assert ([True, True, True, False] == result.relevant).all()

    def test_respects_index(self, relevance_table):
        relevance_table_2 = relevance_table.copy()
        relevance_table_2.reindex(reversed(relevance_table.index))

        result = combine_relevance_tables([relevance_table, relevance_table_2])

        assert ([True, False, True, False] == result.relevant).all()

    def test_aggregates_p_value(self, relevance_table):
        relevance_table_2 = relevance_table.copy()
        relevance_table_2.p_value = [0.2, 0.1, 0.4, 0.3]
        result = combine_relevance_tables([relevance_table, relevance_table_2])

        assert (np.array([0.1, 0.1, 0.3, 0.3]) == result.p_value).all()


class TestGetFeatureType:
    def test_binary(self):
        feature = pd.Series([0.0, 1.0, 1.0])
        assert "binary" == get_feature_type(feature)

    def test_constant(self):
        feature = pd.Series([0.0, 0.0, 0.0])
        assert "constant" == get_feature_type(feature)

    def test_real(self):
        feature = pd.Series([0.0, 1.0, 2.0])
        assert "real" == get_feature_type(feature)
