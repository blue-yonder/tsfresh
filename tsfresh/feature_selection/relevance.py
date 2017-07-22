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
    relevance_table['type'] = [get_feature_type(X[feature]) for feature in relevance_table.index]
    table_real = relevance_table[relevance_table.type == 'real'].copy()
    table_binary = relevance_table[relevance_table.type == 'binary'].copy()

    table_const = relevance_table[relevance_table.type == 'constant'].copy()
    table_const['p_value'] = np.NaN
    table_const['relevant'] = False

    if ml_task == 'classification':
        _calculate = partial(_calculate_relevance_table_for_binary_target, table_real, table_binary, X,
                             test_for_binary_target_real_feature=test_for_binary_target_real_feature,
                             hypotheses_independent=hypotheses_independent, fdr_level=fdr_level)
        unique_labels = y.unique()
        relevance_tables = map_function(_calculate, [(y == label) for label in unique_labels])
        relevance_tables_with_label = zip(unique_labels, relevance_tables)
        relevance_table = combine_relevance_tables(relevance_tables_with_label)
    elif ml_task == 'regression':
        table_real['p_value'] = [target_real_feature_real_test(X[feature], y) for feature in table_real.index]
        table_binary['p_value'] = [target_real_feature_binary_test(X[feature], y) for feature in table_binary.index]
        relevance_table = pd.concat([table_real, table_binary])
        relevance_table = benjamini_hochberg_test(relevance_table, hypotheses_independent, fdr_level)

    relevance_table = pd.concat([relevance_table, table_const], axis=0)
    return relevance_table


def _calculate_relevance_table_for_binary_target(table_real, table_binary, X, y, test_for_binary_target_real_feature,
                                                 hypotheses_independent, fdr_level):
    table_real['p_value'] = [target_binary_feature_real_test(X[feature], y, test_for_binary_target_real_feature)
                             for feature in table_real.index]
    table_binary['p_value'] = [target_binary_feature_binary_test(X[feature], y) for feature in table_binary.index]
    relevance_table = pd.concat([table_real, table_binary])
    return benjamini_hochberg_test(relevance_table, hypotheses_independent, fdr_level)


def check_fs_sig_bh(X, y, target_is_binary,
                    n_jobs=defaults.N_PROCESSES,
                    chunksize=defaults.CHUNKSIZE,
                    fdr_level=defaults.FDR_LEVEL,
                    hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT,
                    test_for_binary_target_real_feature=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE):
    """
    The wrapper function that calls the significance test functions in this package.
    In total, for each feature from the input pandas.DataFrame an univariate feature significance test is conducted.
    Those tests generate p values that are then evaluated by the Benjamini Hochberg procedure to decide which features
    to keep and which to delete.

    We are testing
    
        :math:`H_0` = the Feature is not relevant and can not be added

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

    :param X: The DataFrame containing all the features and the target
    :type X: pandas.DataFrame

    :param y: The target vector
    :type y: pandas.Series

    :param test_for_binary_target_real_feature: Which test to be used for binary target, real feature
    :type test_for_binary_target_real_feature: str

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
            "relevant" (True if the Benjamini Hochberg procedure rejected the null hypothesis for this feature)
    :rtype: pandas.DataFrame

    """
    # Only allow entries for which the target is known!
    y = y.astype(np.float)
    X = X.copy().loc[~(y == np.NaN), :]

    # Create the DataFrame df_features containing the information about the different hypotheses
    # Every row contains information over one feature column from X
    df_features = pd.DataFrame()

    df_features['Feature'] = list(set(X.columns))
    df_features = df_features.set_index('Feature', drop=False)

    # Add relevant columns to df_features
    df_features["relevant"] = np.nan
    df_features["type"] = np.nan
    df_features["p_value"] = np.nan

    # Calculate the feature significance in parallel
    pool = Pool(n_jobs)

    # Helper function which wrapps the _calculate_p_value with many arguments already set
    f = partial(_calculate_p_value, y=y,
                target_is_binary=target_is_binary,
                test_for_binary_target_real_feature=test_for_binary_target_real_feature)
    results = pool.map(f, [X[feature] for feature in df_features['Feature']], chunksize=chunksize)
    p_values_of_features = pd.DataFrame(results)
    df_features.update(p_values_of_features)

    pool.close()
    pool.join()

    # Perform the real feature rejection
    if "const" in set(df_features.type):
        df_features_bh = benjamini_hochberg_test(df_features.loc[~(df_features.type == "const")],
                                                 hypotheses_independent, fdr_level)
        df_features = pd.concat([df_features_bh, df_features.loc[df_features.type == "const"]])
    else:
        df_features = benjamini_hochberg_test(df_features, hypotheses_independent, fdr_level)
        
    # It is very important that we have a boolean "relevant" column, so we do a cast here to be sure
    df_features["relevant"] = df_features["relevant"].astype("bool")

    if defaults.WRITE_SELECTION_REPORT:
        # Write results of BH - Test to file
        if not os.path.exists(defaults.RESULT_DIR):
            os.mkdir(defaults.RESULT_DIR)

        with open(os.path.join(defaults.RESULT_DIR, "fs_bh_results.txt"), 'w') as file_out:
            file_out.write(("Performed BH Test to control the false discovery rate(FDR); \n"
                            "FDR-Level={0};Hypothesis independent={1}\n"
                            ).format(fdr_level, hypotheses_independent))
            df_features.to_csv(index=False, path_or_buf=file_out, sep=';', float_format='%.4f')
    return df_features


def _calculate_p_value(feature_column, y, target_is_binary, test_for_binary_target_real_feature):
    """
    Internal helper function to calculate the p-value of a given feature using one of the dedicated
    functions target_*_feature_*_test.

    :param feature_column: the feature column.
    :type feature_column: pandas.Series

    :param y: the binary target vector
    :type y: pandas.Series

    :param target_is_binary: Whether the target is binary or not
    :type target_is_binary: bool

    :param test_for_binary_target_real_feature: The significance test to be used for binary target and real valued
                                                features. Either ``'mann'`` for the Mann-Whitney-U test or ``'smir'``
                                                for the Kolmogorov-Smirnov test.
    :type test_for_binary_target_real_feature: str

    :return: the p-value of the feature significance test and the type of the tested feature as a Series.
             Lower p-values indicate a higher feature significance.
    :rtype: pd.Series
    """
    # Do not process constant features
    if len(pd.unique(feature_column.values)) == 1:
        _logger.warning("[test_feature_significance] Feature {} is constant".format(feature_column.name))
        return pd.Series({"type": "const", "relevant": False}, name=feature_column.name)

    else:
        if target_is_binary:
            # Decide if the current feature is binary or not
            if len(set(feature_column.values)) == 2:
                type = "binary"
                p_value = target_binary_feature_binary_test(feature_column, y)
            else:
                type = "real"
                p_value = target_binary_feature_real_test(feature_column, y, test_for_binary_target_real_feature)
        else:
            # Decide if the current feature is binary or not
            if len(set(feature_column.values)) == 2:
                type = "binary"
                p_value = target_real_feature_binary_test(feature_column, y)
            else:
                type = "real"
                p_value = target_real_feature_real_test(feature_column, y)

        return pd.Series({"p_value": p_value, "type": type}, name=feature_column.name)


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


def combine_relevance_tables(relevance_tables_with_label):
    """
    Create a combined relevance table out of a list of tuples consisting of a target label
    and its corresponding relevance table.
    The combined relevance table containing the p_values for all target labels.

    :param relevance_tables_with_label: A list of tuples: label, relevance table
    :type relevance_tables_with_label: List[Tuple[Any, pd.DataFrame]]
    :return: The combined relevance table
    :rtype: pandas.DataFrame
    """
    def _append_label_to_p_value_column(a):
        label, df = a
        return df.rename(columns={'p_value': 'p_value_{}'.format(label)})

    def _combine(a, b):
        a.relevant |= b.relevant
        p_value_column = [col for col in b.columns if col.startswith('p_value_')][0]
        return a.join(b[p_value_column])

    relevance_tables = map(_append_label_to_p_value_column, relevance_tables_with_label)
    relevance_table = reduce(_combine, relevance_tables)
    p_value_columns = [col for col in relevance_table.columns if col.startswith('p_value_')]
    relevance_table['p_value'] = relevance_table[p_value_columns].values.min(axis=1)
    return relevance_table


def get_feature_type(feature_column):
    """
    For a given feature, determine if it is real, binary or constant.
    Here binary means that only two unique values occur in the feature.

    :param feature_column: The feature column
    :type feature_column: pandas.Series
    :return: 'constant', 'binary' or 'real'
    """
    n_unique_values = len(feature_column.unique())
    if n_unique_values == 1:
        _logger.warning("[test_feature_significance] Feature {} is constant".format(feature_column.name))
        return 'constant'
    elif n_unique_values == 2:
        return 'binary'
    else:
        return 'real'
