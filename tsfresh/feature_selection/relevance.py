# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
Contains a feature selection method that evaluates the importance of the different extracted features. To do so,
for every feature the influence on the target is evaluated by an univariate tests and the p-Value is calculated.
The methods that calculate the p-values are called feature selectors.

Afterwards the Benjamini Hochberg procedure which is a multiple testing procedure decides which features to keep and
which to cut off (solely based on the p-values).
"""

from __future__ import absolute_import, division, print_function

import logging
from multiprocessing import Pool

import numpy as np
import os
import pandas as pd
from functools import partial, reduce

from tsfresh import defaults
from tsfresh.feature_selection.benjamini_hochberg_test import benjamini_hochberg_test
from tsfresh.feature_selection.significance_tests import target_binary_feature_real_test, \
    target_real_feature_binary_test, target_real_feature_real_test, target_binary_feature_binary_test

_logger = logging.getLogger(__name__)


def calculate_relevance_table(X, y, ml_task='auto', n_jobs=defaults.N_PROCESSES, chunksize=defaults.CHUNKSIZE,
                   test_for_binary_target_binary_feature=defaults.TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
                   test_for_binary_target_real_feature=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE,
                   test_for_real_target_binary_feature=defaults.TEST_FOR_REAL_TARGET_BINARY_FEATURE,
                   test_for_real_target_real_feature=defaults.TEST_FOR_REAL_TARGET_REAL_FEATURE,
                   fdr_level=defaults.FDR_LEVEL, hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT):
    """
    Calculate the relevance table for the features contained in feature matrix `X` with respect to target vector `y`.
    The relevance table is calculated for the intended machine learning task `ml_task`.

    To accomplish this for each feature from the input pandas.DataFrame an univariate feature significance test
    is conducted. Those tests generate p values that are then evaluated by the Benjamini Hochberg procedure to
    decide which features to keep and which to delete.

    We are testing

        :math:`H_0` = the Feature is not relevant and should not be added

    against

        :math:`H_1` = the Feature is relevant and should be kept

    or in other words

        :math:`H_0` = Target and Feature are independent / the Feature has no influence on the target

        :math:`H_1` = Target and Feature are associated / dependent

    When the target is binary this becomes

        :math:`H_0 = \\left( F_{\\text{target}=1} = F_{\\text{target}=0} \\right)`

        :math:`H_1 = \\left( F_{\\text{target}=1} \\neq F_{\\text{target}=0} \\right)`

    Where :math:`F` is the distribution of the target.

    In the same way we can state the hypothesis when the feature is binary

        :math:`H_0 =  \\left( T_{\\text{feature}=1} = T_{\\text{feature}=0} \\right)`

        :math:`H_1 = \\left( T_{\\text{feature}=1} \\neq T_{\\text{feature}=0} \\right)`

    Here :math:`T` is the distribution of the target.

    TODO: And for real valued?

    :param X: Feature matrix in the format mentioned before which will be reduced to only the relevant features.
              It can contain both binary or real-valued features at the same time.
    :type X: pandas.DataFrame

    :param y: Target vector which is needed to test which features are relevant. Can be binary or real-valued.
    :type y: pandas.Series or numpy.ndarray

    :param ml_task: The intended machine learning task. Either `'classification'`, `'regression'` or `'auto'`.
                    Defaults to `'auto'`, meaning the intended task is inferred from `y`.
                    If `y` has a boolean, integer or object dtype, the task is assumend to be classification,
                    else regression.
    :type ml_task: str

    :param test_for_binary_target_binary_feature: Which test to be used for binary target, binary feature (currently unused)
    :type test_for_binary_target_binary_feature: str

    :param test_for_binary_target_real_feature: Which test to be used for binary target, real feature
    :type test_for_binary_target_real_feature: str

    :param test_for_real_target_binary_feature: Which test to be used for real target, binary feature (currently unused)
    :type test_for_real_target_binary_feature: str

    :param test_for_real_target_real_feature: Which test to be used for real target, real feature (currently unused)
    :type test_for_real_target_real_feature: str

    :param fdr_level: The FDR level that should be respected, this is the theoretical expected percentage of irrelevant
                      features among all created features.
    :type fdr_level: float

    :param hypotheses_independent: Can the significance of the features be assumed to be independent?
                                   Normally, this should be set to False as the features are never
                                   independent (e.g. mean and median)
    :type hypotheses_independent: bool

    :param n_jobs: Number of processes to use during the p-value calculation
    :type n_jobs: int

    :param chunksize: Size of the chunks submitted to the worker processes
    :type chunksize: int

    :return: A pandas.DataFrame with each column of the input DataFrame X as index with information on the significance
             of this particular feature. The DataFrame has the columns
             "Feature",
             "type" (binary, real or const),
             "p_value" (the significance of this feature as a p-value, lower means more significant)
             "relevant" (True if the Benjamini Hochberg procedure rejected the null hypothesis [the feature is
             not relevant] for this feature)
    :rtype: pandas.DataFrame
    """
    if ml_task not in ['auto', 'classification', 'regression']:
        raise ValueError('ml_task must be one of: \'auto\', \'classification\', \'regression\'')
    elif ml_task == 'auto':
        ml_task = infer_ml_task(y)

    if n_jobs == 0:
        map_function = map
    else:
        pool = Pool(n_jobs)
        map_function = partial(pool.map, chunksize=chunksize)

    relevance_table = pd.DataFrame(index=pd.Series(X.columns, name='feature'))
    relevance_table['feature'] = relevance_table.index
    relevance_table['type'] = pd.Series(
        map_function(get_feature_type, [X[feature] for feature in relevance_table.index]),
        index=relevance_table.index
    )
    table_real = relevance_table[relevance_table.type == 'real'].copy()
    table_binary = relevance_table[relevance_table.type == 'binary'].copy()

    table_const = relevance_table[relevance_table.type == 'constant'].copy()
    table_const['p_value'] = np.NaN
    table_const['relevant'] = False

    if len(table_const) == len(relevance_table):
        return table_const

    if ml_task == 'classification':
        tables = []
        for label in y.unique():
            _test_real_feature = partial(target_binary_feature_real_test, y=(y == label),
                                              test=test_for_binary_target_real_feature)
            _test_binary_feature = partial(target_binary_feature_binary_test, y=(y == label))
            tmp = _calculate_relevance_table_for_implicit_target(
                table_real, table_binary, X, _test_real_feature, _test_binary_feature, hypotheses_independent,
                fdr_level, map_function
            )
            tables.append(tmp)
        relevance_table = combine_relevance_tables(tables)
    elif ml_task == 'regression':
        _test_real_feature = partial(target_real_feature_real_test, y=y)
        _test_binary_feature = partial(target_real_feature_binary_test, y=y)
        relevance_table = _calculate_relevance_table_for_implicit_target(
            table_real, table_binary, X, _test_real_feature, _test_binary_feature, hypotheses_independent, fdr_level,
            map_function
        )

    relevance_table = pd.concat([relevance_table, table_const], axis=0)

    if n_jobs != 0:
        pool.close()
        pool.terminate()
        pool.join()

    if sum(relevance_table['relevant']) == 0:
        _logger.warning("No feature was found relevant for {} for fdr level = {}. "
                        "Consider using a lower fdr level or other features.".format(ml_task, fdr_level))

    return relevance_table


def _calculate_relevance_table_for_implicit_target(table_real, table_binary, X, test_real_feature, test_binary_feature,
                                                   hypotheses_independent, fdr_level, map_function):
    table_real['p_value'] = pd.Series(
        map_function(test_real_feature, [X[feature] for feature in table_real.index]),
        index=table_real.index
    )
    table_binary['p_value'] = pd.Series(
        map_function(test_binary_feature, [X[feature] for feature in table_binary.index]),
        index=table_binary.index
    )
    relevance_table = pd.concat([table_real, table_binary])
    return benjamini_hochberg_test(relevance_table, hypotheses_independent, fdr_level)


def infer_ml_task(y):
    """
    Infer the machine learning task to select for.
    The result will be either `'regression'` or `'classification'`.
    If the target vector only consists of integer typed values or objects, we assume the task is `'classification'`.
    Else `'regression'`.

    :param y: The target vector y.
    :type y: pandas.Series
    :return: 'classification' or 'regression'
    :rtype: str
    """
    if y.dtype.kind in np.typecodes['AllInteger'] or y.dtype == np.object:
        ml_task = 'classification'
    else:
        ml_task = 'regression'

    _logger.warning('Infered {} as machine learning task'.format(ml_task))
    return ml_task


def combine_relevance_tables(relevance_tables):
    """
    Create a combined relevance table out of a list of relevance tables,
    aggregating the p-values and the relevances.

    :param relevance_tables: A list of relevance tables
    :type relevance_tables: List[pd.DataFrame]
    :return: The combined relevance table
    :rtype: pandas.DataFrame
    """
    def _combine(a, b):
        a.relevant |= b.relevant
        a.p_value = a.p_value.combine(b.p_value, min, 1)
        return a

    return reduce(_combine, relevance_tables)


def get_feature_type(feature_column):
    """
    For a given feature, determine if it is real, binary or constant.
    Here binary means that only two unique values occur in the feature.

    :param feature_column: The feature column
    :type feature_column: pandas.Series
    :return: 'constant', 'binary' or 'real'
    """
    n_unique_values = len(set(feature_column.values))
    if n_unique_values == 1:
        _logger.warning("[test_feature_significance] Feature {} is constant".format(feature_column.name))
        return 'constant'
    elif n_unique_values == 2:
        return 'binary'
    else:
        return 'real'
