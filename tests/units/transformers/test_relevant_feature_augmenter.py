# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import mock
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from tests.fixtures import DataTestCase, warning_free
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.transformers.relevant_feature_augmenter import RelevantFeatureAugmenter


class RelevantFeatureAugmenterTestCase(DataTestCase):
    def setUp(self):
        self.test_df = self.create_test_data_sample()
        fc_parameters = {"length": None}
        self.kind_to_fc_parameters = {
            "a": fc_parameters.copy(),
            "b": fc_parameters.copy(),
        }

    def test_not_fitted(self):
        augmenter = RelevantFeatureAugmenter()

        X = pd.DataFrame()

        self.assertRaises(RuntimeError, augmenter.transform, X)

    def test_no_timeseries(self):
        augmenter = RelevantFeatureAugmenter()

        X = pd.DataFrame()
        y = pd.Series(dtype="float64")

        self.assertRaises(RuntimeError, augmenter.fit, X, y)
        self.assertRaises(RuntimeError, augmenter.fit_transform, X, y)

    def test_nothing_relevant(self):
        augmenter = RelevantFeatureAugmenter(
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            column_value="val",
            column_id="id",
            column_sort="sort",
            column_kind="kind",
        )

        y = pd.Series({10: 1, 500: 0})
        X = pd.DataFrame(index=[10, 500])

        augmenter.set_timeseries_container(self.test_df)
        augmenter.fit(X, y)
        transformed_X = augmenter.transform(X.copy())

        fit_transformed_X = augmenter.fit_transform(X, y)

        self.assertEqual(list(transformed_X.columns), [])
        self.assertEqual(list(transformed_X.index), list(X.index))
        self.assertEqual(list(fit_transformed_X.columns), [])
        self.assertEqual(list(fit_transformed_X.index), list(X.index))

    def test_filter_only_tsfresh_features_true(self):
        """
        The boolean flag `filter_only_tsfresh_features` makes sure that only the time series based features are
        filtered. This unit tests checks that
        """

        augmenter = RelevantFeatureAugmenter(
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            filter_only_tsfresh_features=True,
            column_value="val",
            column_id="id",
            column_sort="sort",
            column_kind="kind",
        )

        y = pd.Series({10: 1, 500: 0})
        X = pd.DataFrame(index=[10, 500])
        X["pre_feature"] = 0

        augmenter.set_timeseries_container(self.test_df)
        augmenter.fit(X, y)
        transformed_X = augmenter.transform(X.copy())

        fit_transformed_X = augmenter.fit_transform(X, y)

        self.assertEqual(
            sum(["pre_feature" == column for column in transformed_X.columns]), 1
        )
        self.assertEqual(
            sum(["pre_feature" == column for column in fit_transformed_X.columns]), 1
        )

    def test_filter_only_tsfresh_features_false(self):
        """
        The boolean flag `filter_only_tsfresh_features` makes sure that only the time series based features are
        filtered. This unit tests checks that
        """

        augmenter = RelevantFeatureAugmenter(
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            filter_only_tsfresh_features=False,
            column_value="val",
            column_id="id",
            column_sort="sort",
            column_kind="kind",
        )

        df, y = self.create_test_data_sample_with_target()
        X = pd.DataFrame(index=np.unique(df.id))
        X["pre_drop"] = 0
        X["pre_keep"] = y

        augmenter.set_timeseries_container(df)
        augmenter.fit(X, y)
        transformed_X = augmenter.transform(X.copy())

        fit_transformed_X = augmenter.fit_transform(X, y)

        self.assertEqual(
            sum(["pre_keep" == column for column in transformed_X.columns]), 1
        )
        self.assertEqual(
            sum(["pre_drop" == column for column in transformed_X.columns]), 0
        )
        self.assertEqual(
            sum(["pre_keep" == column for column in fit_transformed_X.columns]), 1
        )
        self.assertEqual(
            sum(["pre_drop" == column for column in fit_transformed_X.columns]), 0
        )

    @mock.patch("tsfresh.transformers.feature_selector.calculate_relevance_table")
    def test_does_impute(self, calculate_relevance_table_mock):
        df = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=["id", "time", "value"])
        X = pd.DataFrame(index=[1])
        y = pd.Series([0, 1])
        fc_parameters = {"autocorrelation": [{"lag": 2}]}

        calculate_relevance_table_mock.return_value = pd.DataFrame(
            columns=["feature", "p_value", "relevant"]
        )
        augmenter = RelevantFeatureAugmenter(
            column_id="id", column_sort="time", default_fc_parameters=fc_parameters
        )
        augmenter.set_timeseries_container(df)
        with warning_free():
            augmenter.fit(X, y)

        assert calculate_relevance_table_mock.call_count == 1
        assert not calculate_relevance_table_mock.call_args[0][0].isnull().any().any()

    def test_no_ids_present(self):
        augmenter = RelevantFeatureAugmenter(
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            filter_only_tsfresh_features=False,
            column_value="val",
            column_id="id",
            column_sort="sort",
            column_kind="kind",
        )

        df, y = self.create_test_data_sample_with_target()
        X_with_wrong_ids = pd.DataFrame(index=[-999])

        augmenter.set_timeseries_container(df)

        self.assertRaisesRegex(
            AttributeError,
            r"The ids of the time series container",
            augmenter.fit,
            X_with_wrong_ids,
            y,
        )
        self.assertRaisesRegex(
            AttributeError,
            r"The ids of the time series container",
            augmenter.fit_transform,
            X_with_wrong_ids,
            y,
        )

    def test_multiclass_selection(self):
        augmenter = RelevantFeatureAugmenter(
            column_value="val",
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            multiclass=True,
            n_significant=3,
        )

        df, y = self.create_test_data_sample_with_multiclass_target()
        X = pd.DataFrame(index=np.unique(df.id))

        augmenter.set_timeseries_container(df)
        fit_transformed_X = augmenter.fit_transform(X, y)

        self.assertEqual(len(fit_transformed_X.columns), 4)


def test_relevant_augmentor_cross_validated():
    """
    Validates that the RelevantFeatureAugmenter can be cloned in pipelines, see issue 537
    """
    n = 16  # number of samples, needs to be divisable by 4
    index = range(n)
    df_ts = pd.DataFrame(
        {
            "time": [10, 11] * n,
            "id": np.repeat(index, 2),
            "value": [0, 1] * (n // 4)
            + [1, 2] * (n // 4)
            + [10, 11] * (n // 4)  # class 0
            + [12, 14] * (n // 4),
        }
    )
    y = pd.Series(data=[0] * (n // 2) + [1] * (n // 2), index=index)
    X = pd.DataFrame(index=index)
    augmenter = RelevantFeatureAugmenter(
        column_id="id",
        column_sort="time",
        timeseries_container=df_ts,
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True,
        show_warnings=False,
        fdr_level=0.90,
    )
    pipeline = Pipeline(
        [
            ("augmenter", augmenter),
            ("classifier", RandomForestClassifier(random_state=1)),
        ]
    )

    scores = model_selection.cross_val_score(pipeline, X, y, cv=2)
    assert (scores == np.array([1, 1])).all()
