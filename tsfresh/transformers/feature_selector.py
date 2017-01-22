# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import pandas as pd
from tsfresh import defaults
from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh.feature_selection.feature_selector import check_fs_sig_bh


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible estimator, for reducing the number of features in a dataset to only those,
    that are relevant and significant to a given target. It is basically a wrapper around
    :func:`~tsfresh.feature_selection.feature_selector.check_fs_sig_bh`.

    The check is done by testing the hypothesis

        :math:`H_0` = the Feature is not relevant and can not be added`

    against

        :math:`H_1` = the Feature is relevant and should be kept

    using several statistical tests (depending on whether the feature or/and the target is binary
    or not). Using the Benjamini Hochberg procedure, only features in :math:`H_0` are rejected.

    This estimator - as most of the sklearn estimators - works in a two step procedure. First, it is fitted
    on training data, where the target is known:

    >>> import pandas as pd
    >>> X_train, y_train = pd.DataFrame(), pd.Series() # fill in with your features and target
    >>> from tsfresh.transformers import FeatureSelector
    >>> selector = FeatureSelector()
    >>> selector.fit(X_train, y_train)

    In this example the list of relevant features is empty:
    >>> selector.relevant_features
    >>> []

    The same holds for the feature importance:
    >>> selector.feature_importances_
    >>> array([], dtype=float64)

    The estimator keeps track on those features, that were relevant in the training step. If you
    apply the estimator after the training, it will delete all other features in the testing
    data sample:

    >>> X_test = pd.DataFrame()
    >>> X_selected = selector.transform(X_test)

    After that, X_selected will only contain the features that were relevant during the training.

    If you are interested in more information on the features, you can look into the member
    ``relevant_features`` after the fit.
    """
    def __init__(self, test_for_binary_target_binary_feature=defaults.TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
                 test_for_binary_target_real_feature=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE,
                 test_for_real_target_binary_feature=defaults.TEST_FOR_REAL_TARGET_BINARY_FEATURE,
                 test_for_real_target_real_feature=defaults.TEST_FOR_REAL_TARGET_REAL_FEATURE,
                 fdr_level=defaults.FDR_LEVEL, hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT,
                 n_processes=defaults.N_PROCESSES, chunksize=defaults.CHUNKSIZE):
        """
        Create a new FeatureSelector instance.

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

        :param n_processes: Number of processes to use during the p-value calculation
        :type n_processes: int

        :param chunksize: Size of the chunks submitted to the worker processes
        :type chunksize: int
        """
        self.relevant_features = None
        self.feature_importances_ = None
        self.p_values = None
        self.features = None

        self.test_for_binary_target_binary_feature = test_for_binary_target_binary_feature
        self.test_for_binary_target_real_feature = test_for_binary_target_real_feature
        self.test_for_real_target_binary_feature = test_for_real_target_binary_feature
        self.test_for_real_target_real_feature = test_for_real_target_real_feature

        self.fdr_level = fdr_level
        self.hypotheses_independent = hypotheses_independent

        self.n_processes = n_processes
        self.chunksize = chunksize

    def fit(self, X, y):
        """
        Extract the information, which of the features are relevent using the given target.

        For more information, please see the :func:`~tsfresh.festure_selection.festure_selector.check_fs_sig_bh`
        function. All columns in the input data sample are treated as feature. The index of all
        rows in X must be present in y.

        :param X: data sample with the features, which will be classified as relevant or not
        :type X: pandas.DataFrame or numpy.array

        :param y: target vecotr to be used, to classify the features
        :type y: pandas.Series or numpy.array

        :return: the fitted estimator with the information, which features are relevant
        :rtype: FeatureSelector
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X.copy())

        if not isinstance(y, pd.Series):
            y = pd.Series(y.copy())

        df_bh = check_fs_sig_bh(X, y, self.n_processes, self.chunksize,
                                self.fdr_level, self.hypotheses_independent,
                                self.test_for_binary_target_real_feature)
        self.relevant_features = df_bh.loc[df_bh.rejected].Feature.tolist()
        self.feature_importances_ = 1.0 - df_bh.p_value.values
        self.p_values = df_bh.p_value.values
        self.features = df_bh.Feature.tolist()

        return self

    def transform(self, X):
        """
        Delete all features, which were not relevant in the fit phase.

        :param X: data sample with all features, which will be reduced to only those that are relevant
        :type X: pandas.DataSeries or numpy.array

        :return: same data sample as X, but with only the relevant features
        :rtype: pandas.DataFrame or numpy.array
        """
        if self.relevant_features is None:
            raise RuntimeError("You have to call fit before.")

        if isinstance(X, pd.DataFrame):
            return X.copy().loc[:, self.relevant_features]
        else:
            return X[:, self.relevant_features]
