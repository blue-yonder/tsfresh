# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import numpy as np

def benjamini_hochberg_test(df_pvalues, hypotheses_independent, fdr_level):
    """
    This is an implementation of the benjamini hochberg procedure [1]_ that determines if the null hypothesis
    for a given feature can be rejected. For this the test regards the features' p-values
    and controls the global false discovery rate, which is the ratio of false rejections by all rejections:

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

    :param hypotheses_independent: Can the significance of the features be assumed to be independent?
                                   Normally, this should be set to False as the features are never
                                   independent (e.g. mean and median)
    :type hypotheses_independent: bool

    :param fdr_level: The FDR level that should be respected, this is the theoretical expected percentage of irrelevant
                      features among all created features.
    :type fdr_level: float

    :return: The same DataFrame as the input, but with an added boolean column "relevant"
             denoting if the null hypotheses has been rejected for a given feature.
    :rtype: pandas.DataFrame
    """

    # Get auxiliary variables and vectors
    df_pvalues = df_pvalues.sort_values(by="p_value")
    m = len(df_pvalues)
    K = np.arange(1, m + 1)

    # Calculate the weight vector C
    if hypotheses_independent:
        # c(k) = 1
        C = np.ones(m)
    else:
        # c(k) = \sum_{i=1}^m 1/i
        C = np.cumsum(1.0 / K)

    # Calculate the vector T to compare to the p_value
    T = (fdr_level * K) / (m * C)

    # Get the last p_value for which H0 has been rejected
    try:
        k_max = list(df_pvalues.p_value <= T).index(False)
    except ValueError:
        k_max = m

    # Add the column denoting if null hypothesis has been rejected
    df_pvalues["relevant"] = [True] * k_max + [False] * (m - k_max)

    return df_pvalues