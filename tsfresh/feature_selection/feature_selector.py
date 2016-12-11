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

from functools import partial

from builtins import zip
from builtins import range
import os
import numpy as np
import pandas as pd
import logging
from multiprocessing import Pool
from tsfresh.feature_selection.significance_tests import target_binary_feature_real_test, \
    target_real_feature_binary_test, target_real_feature_real_test, target_binary_feature_binary_test

from tsfresh.feature_selection.settings import FeatureSignificanceTestsSettings


_logger = logging.getLogger(__name__)


def check_fs_sig_bh(X, y, settings=None):
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

    :param settings: The feature selection settings to use for performing the tests.
    :type settings: FeatureSignificanceTestsSettings

    :return: A pandas.DataFrame with each column of the input DataFrame X as index with information on the significance
            of this particular feature. The DataFrame has the columns
            "Feature",
            "type" (binary, real or const),
            "p_value" (the significance of this feature as a p-value, lower means more significant)
            "rejected" (if the Benjamini Hochberg procedure rejected this feature)
    :rtype: pandas.DataFrame

    """
    if settings is None:
        settings = FeatureSignificanceTestsSettings()

    target_is_binary = len(set(y)) == 2

    # todo: solve the multiclassification case. for a multi classification the algorithm considers the target to be
    # regression. Instead one could perform a binary one versus all classification.

    # Only allow entries for which the target is known!
    y = y.astype(np.float)
    X = X.copy().loc[~(y == np.NaN), :]

    # Create the DataFrame df_features containing the information about the different hypotheses
    # Every row contains information over one feature column from X
    df_features = pd.DataFrame()

    df_features['Feature'] = list(set(X.columns))
    df_features = df_features.set_index('Feature', drop=False)

    # Add relevant columns to df_features
    df_features["rejected"] = np.nan
    df_features["type"] = np.nan
    df_features["p_value"] = np.nan

    # Calculate the feature significance in parallel
    pool = Pool(settings.n_processes)

    # Helper function which wrapps the _calculate_p_value with many arguments already set
    f = partial(_calculate_p_value, y=y, settings=settings, target_is_binary=target_is_binary)
    results = pool.map(f, [X[feature] for feature in df_features['Feature']], chunksize=settings.chunksize)
    p_values_of_features = pd.DataFrame(results)
    df_features.update(p_values_of_features)

    pool.close()
    pool.join()

    # Perform the real feature rejection
    if "const" in set(df_features.type):
        df_features_bh = benjamini_hochberg_test(df_features.loc[~(df_features.type == "const")], settings)
        df_features = pd.concat([df_features_bh, df_features.loc[df_features.type == "const"]])
    else:
        df_features = benjamini_hochberg_test(df_features, settings)
        
    # It is very important that we have a boolean "rejected" column, so we do a cast here to be sure
    df_features["rejected"] = df_features["rejected"].astype("bool")

    if settings.write_selection_report:
        # Write results of BH - Test to file
        if not os.path.exists(settings.result_dir):
            os.mkdir(settings.result_dir)

        with open(os.path.join(settings.result_dir, "fs_bh_results.txt"), 'w') as file_out:
            file_out.write(("Performed BH Test to control the false discovery rate(FDR); \n"
                            "FDR-Level={0};Hypothesis independent={1}\n"
                            ).format(settings.fdr_level, settings.hypotheses_independent))
            df_features.to_csv(index=False, path_or_buf=file_out, sep=';', float_format='%.4f')
    return df_features


def _calculate_p_value(feature_column, y, settings, target_is_binary):
    """
    Internal helper function to calculate the p-value of a given feature using one of the dedicated
    functions target_*_feature_*_test.

    :param feature_column: the feature column.
    :type feature_column: pandas.Series

    :param y: the binary target vector
    :type y: pandas.Series

    :param settings: The settings object to control how the significance is calculated.
    :type settings: FeatureSignificanceTestsSettings

    :param target_is_binary: Whether the target is binary or not
    :type target_is_binary: bool

    :return: the p-value of the feature significance test and the type of the tested feature as a Series.
             Lower p-values indicate a higher feature significance.
    :rtype: pd.Series
    """
    # Do not process constant features
    if len(pd.unique(feature_column.values)) == 1:
        _logger.warning("[test_feature_significance] Feature {} is constant".format(feature_column.name))
        return pd.Series({"type": "const", "rejected": False}, name=feature_column.name)

    else:
        if target_is_binary:
            # Decide if the current feature is binary or not
            if len(set(feature_column.values)) == 2:
                type = "binary"
                p_value = target_binary_feature_binary_test(feature_column, y, settings)
            else:
                type = "real"
                p_value = target_binary_feature_real_test(feature_column, y, settings)
        else:
            # Decide if the current feature is binary or not
            if len(set(feature_column.values)) == 2:
                type = "binary"
                p_value = target_real_feature_binary_test(feature_column, y, settings)
            else:
                type = "real"
                p_value = target_real_feature_real_test(feature_column, y, settings)

        return pd.Series({"p_value": p_value, "type": type}, name=feature_column.name)


def benjamini_hochberg_test(df_pvalues, settings):
    """
    This is an implementation of the benjamini hochberg procedure that calculates which of the hypotheses belonging
    to the different p-Values from df_p to reject. While doing so, this test controls the false discovery rate,
    which is the ratio of false rejections by all rejections:

    .. math::

        FDR = \\mathbb{E} \\left [ \\frac{ |\\text{false rejections}| }{ |\\text{all rejections}|} \\right]


    References
    ----------

    .. [1] Benjamini, Yoav and Yekutieli, Daniel (2001).
        The control of the false discovery rate in multiple testing under dependency.
        Annals of statistics, 1165--1188


    :param df_pvalues: This DataFrame should contain the p_values of the different hypotheses in a column named
                       "p_values".
    :type df_pvalues: pandas.DataFrame

    :param settings: The settings object to use for controlling the false discovery rate (FDR_level) and
           whether to threat the hypothesis independent or not (hypotheses_independent).
    :type settings: FeatureSignificanceTestsSettings

    :return: The same DataFrame as the input, but with an added boolean column "rejected".
    :rtype: pandas.DataFrame
    """

    # Get auxiliary variables and vectors
    df_pvalues = df_pvalues.sort_values(by="p_value")
    m = len(df_pvalues)
    K = list(range(1, m + 1))

    # Calculate the weight vector C
    if settings.hypotheses_independent:
        # c(k) = 1
        C = [1] * m
    else:
        # c(k) = \sum_{i=1}^m 1/i
        C = [sum([1.0 / i for i in range(1, k + 1)]) for k in K]

    # Calculate the vector T to compare to the p_value
    T = [settings.fdr_level * k / m * 1.0 / c for k, c in zip(K, C)]

    # Get the last rejected p_value
    try:
        k_max = list(df_pvalues.p_value <= T).index(False)
    except ValueError:
        k_max = m

    # Add the column denoting if hypothesis was rejected
    df_pvalues["rejected"] = [True] * k_max + [False] * (m - k_max)

    return df_pvalues
