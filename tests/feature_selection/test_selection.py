# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import numpy as np
import pandas as pd
import pytest
from future.utils import lrange

from tsfresh.feature_selection.selection import select_features
from tsfresh.feature_selection.relevance import infer_ml_task, combine_relevance_tables, calculate_relevance_table


class TestSelectFeatures:
    def test_assert_list(self):
        with pytest.raises(TypeError):
            select_features(pd.DataFrame(index=range(2)),[1,2,3])


    def test_assert_one_row_X(self):
        X = pd.DataFrame([1], index=[1])
        y = pd.Series([1], index=[1])
        with pytest.raises(ValueError):
            select_features(X, y)


    def test_assert_different_index(self):
        X = pd.DataFrame(list(range(3)), index=[1, 2, 3])
        y = pd.Series(range(3), index=[1, 3, 4])
        with pytest.raises(ValueError):
            select_features(X, y)


    def test_assert_shorter_y(self):
        X = pd.DataFrame([1, 2], index=[1, 2])
        y = np.array([1])
        with pytest.raises(ValueError):
            select_features(X, y)


    def test_selects_for_each_class(self):
        df = pd.DataFrame()
        df['f1'] = [10] * 10 + lrange(10) + lrange(10)
        df['f2'] = lrange(10) + [10] * 10 + lrange(10)
        df['f3'] = lrange(10) + lrange(10) + [10] * 10
        df['y'] = [0] * 10 + [1] * 10 + [2] * 10

        y = df.y
        X = df.drop(['y'], axis=1)
        X_relevant = select_features(X, y, ml_task='classification')
        assert {'f1', 'f2', 'f3'} == set(X_relevant.columns)


class TestInferMLTask:
    def test_infers_classification_for_integer_target(self):
        y = pd.Series([1, 2, 3])
        assert 'classification' == infer_ml_task(y)


    def test_infers_classification_for_boolean_target(self):
        y = pd.Series([True, False, False])
        assert 'classification' == infer_ml_task(y)

    def test_infers_classification_for_object_target(self):
        y = pd.Series(['high', 'low'])
        assert 'classification' == infer_ml_task(y)

    def test_infers_regression_for_float_target(self):
        y = pd.Series([1.0, 1.5, 1.7])
        assert 'regression' == infer_ml_task(y)


class TestGetRelevanceTable:
    def test_restrict_ml_task_options(self):
        X = pd.DataFrame(list(range(3)))
        y = pd.Series(range(3))
        with pytest.raises(ValueError):
            calculate_relevance_table(X, y, ml_task='some_other_task')


class TestCombineRelevanceTables:
    @pytest.fixture()
    def relevance_table(self):
        relevance_table = pd.DataFrame()
        relevance_table['Feature'] = ['f1', 'f2', 'f3', 'f4']
        relevance_table['relevant'] = [True, False, True, False]
        relevance_table['type'] = ['real'] * 4
        relevance_table['p_value'] = [0.1, 0.2, 0.3, 0.4]
        return relevance_table

    def test_appends_label_to_p_value_column(self, relevance_table):
        result = combine_relevance_tables([(0, relevance_table)])
        assert 'p_value_0' in result.columns

    def test_disjuncts_relevance(self, relevance_table):
        relevance_table_2 = relevance_table.copy()
        relevance_table_2.relevant = [False, True, True, False]
        result = combine_relevance_tables([(0, relevance_table), (1, relevance_table_2)])

        assert ([True, True, True, False] == result.relevant).all()

    def test_respects_index(self, relevance_table):
        relevance_table_2 = relevance_table.copy()
        relevance_table_2.reindex(reversed(relevance_table.index))

        result = combine_relevance_tables([(0, relevance_table), (1, relevance_table_2)])

        assert ([True, False, True, False] == result.relevant).all()

    def test_preserves_p_values(self, relevance_table):
        relevance_table_2 = relevance_table.copy()
        relevance_table_2.p_value = 1.0 - relevance_table_2.p_value
        result = combine_relevance_tables([(0, relevance_table), (1, relevance_table_2)])

        assert (relevance_table.p_value == result.p_value_0).all()
        assert (relevance_table_2.p_value == result.p_value_1).all()

    def test_aggregates_p_value(self, relevance_table):
        relevance_table_2 = relevance_table.copy()
        relevance_table_2.p_value = [0.2, 0.1, 0.4, 0.3]
        result = combine_relevance_tables([(0, relevance_table), (1, relevance_table_2)])
        
        assert (np.array([0.1, 0.1, 0.3, 0.3]) == result.p_value).all()
