# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import os
from multiprocessing import cpu_count
from builtins import object


class FeatureSignificanceTestsSettings(object):
    """
    The settings object for controlling the feature significance tests.
    Normally, you do not have to handle these settings on your own, as the chosen defaults are quite sensible.

    This object is passed to mostly all functions in the feature_selection submodules.

    If you want non-default settings, create a new settings object and pass it to the functions, for example if you
    want a less conservative selection of features you could increase the fdr level to 10%.

    >>> from tsfresh.feature_selection import FeatureSignificanceTestsSettings
    >>> settings = FeatureSignificanceTestsSettings()
    >>> settings.fdr_level = 0.1
    >>> from tsfresh.feature_selection import select_features
    >>> select_features(X, y, feature_selection_settings=settings)

    This selection process will return more features as the fdr level was raised.

    """

    def __init__(self):
        """
        Create a new settings object with the default arguments.
        """
        #: Which test to be used for binary target, binary feature (unused)
        self.test_for_binary_target_binary_feature = "fisher"
        #: Which test to be used for binary target, real feature
        self.test_for_binary_target_real_feature = "mann"
        #: Which test to be used for real target, binary feature (unused)
        self.test_for_real_target_binary_feature = "mann"
        #: Which test to be used for real target, real feature (unused)
        self.test_for_real_target_real_feature = "kendall"

        #: The FDR level that should be respected, this is the theoretical expected percentage of irrelevant features
        #: among all created features. E.g.
        self.fdr_level = 0.05

        #: Can the significance of the features be assumed to be independent?
        #: Normally, this should be set to False as the features are never independent (think about mean and median)
        self.hypotheses_independent = False

        #: Whether to store the selection report after the Benjamini Hochberg procedure has finished.
        self.write_selection_report = False

        #: Where to store the selection import
        self.result_dir = "logging"

        #: Number of processes to use during the p-value calculation
        self.n_processes = int(os.getenv("NUMBER_OF_CPUS") or cpu_count())

        # Size of the chunks submitted to the worker processes
        self.chunksize = None
