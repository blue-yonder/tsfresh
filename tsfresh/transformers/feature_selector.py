# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from tsfresh import defaults
from tsfresh.feature_selection.relevance import calculate_relevance_table


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

    def __init__(
        self,
        test_for_binary_target_binary_feature=defaults.TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
        test_for_binary_target_real_feature=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE,
        test_for_real_target_binary_feature=defaults.TEST_FOR_REAL_TARGET_BINARY_FEATURE,
        test_for_real_target_real_feature=defaults.TEST_FOR_REAL_TARGET_REAL_FEATURE,
        fdr_level=defaults.FDR_LEVEL,
        hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT,
        n_jobs=defaults.N_PROCESSES,
        chunksize=defaults.CHUNKSIZE,
        ml_task="auto",
        multiclass=False,
        n_significant=1,
        multiclass_p_values="min",
    ):
        """
        Create a new FeatureSelector instance.

        :param test_for_binary_target_binary_feature: Which test to be used for binary target, binary feature
                                                      (currently unused)
        :type test_for_binary_target_binary_feature: str

        :param test_for_binary_target_real_feature: Which test to be used for binary target, real feature
        :type test_for_binary_target_real_feature: str

        :param test_for_real_target_binary_feature: Which test to be used for real target, binary feature
                                                    (currently unused)
        :type test_for_real_target_binary_feature: str

        :param test_for_real_target_real_feature: Which test to be used for real target, real feature (currently unused)
        :type test_for_real_target_real_feature: str

        :param fdr_level: The FDR level that should be respected, this is the theoretical expected percentage
                          of irrelevant features among all created features.
        :type fdr_level: float

        :param hypotheses_independent: Can the significance of the features be assumed to be independent?
                                       Normally, this should be set to False as the features are never
                                       independent (e.g. mean and median)
        :type hypotheses_independent: bool

        :param n_jobs: Number of processes to use during the p-value calculation
        :type n_jobs: int

        :param chunksize: Size of the chunks submitted to the worker processes
        :type chunksize: int

        :param ml_task: The intended machine learning task. Either `'classification'`, `'regression'` or `'auto'`.
                    Defaults to `'auto'`, meaning the intended task is inferred from `y`.
                    If `y` has a boolean, integer or object dtype, the task is assumed to be classification,
                    else regression.
        :type ml_task: str

        :param multiclass: Whether the problem is multiclass classification. This modifies the way in which features
                       are selected. Multiclass requires the features to be statistically significant for
                       predicting n_significant classes.
        :type multiclass: bool

        :param n_significant: The number of classes for which features should be statistically significant predictors
                            to be regarded as 'relevant'
        :type n_significant: int

        :param multiclass_p_values: The desired method for choosing how to display multiclass p-values for each feature.
                                    Either `'avg'`, `'max'`, `'min'`, `'all'`. Defaults to `'min'`, meaning the p-value
                                    with the highest significance is chosen. When set to `'all'`, the attributes
                                    `self.feature_importances_` and `self.p_values` are of type pandas.DataFrame, where
                                    each column corresponds to a target class.
        :type multiclass_p_values: str
        """
        self.relevant_features = None
        self.feature_importances_ = None
        self.p_values = None
        self.features = None

        self.test_for_binary_target_binary_feature = (
            test_for_binary_target_binary_feature
        )
        self.test_for_binary_target_real_feature = test_for_binary_target_real_feature
        self.test_for_real_target_binary_feature = test_for_real_target_binary_feature
        self.test_for_real_target_real_feature = test_for_real_target_real_feature

        self.fdr_level = fdr_level
        self.hypotheses_independent = hypotheses_independent

        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.ml_task = ml_task
        self.multiclass = multiclass
        self.n_significant = n_significant
        self.multiclass_p_values = multiclass_p_values

    def fit(self, X, y):
        """
        Extract the information, which of the features are relevant using the given target.

        For more information, please see the :func:`~tsfresh.festure_selection.festure_selector.check_fs_sig_bh`
        function. All columns in the input data sample are treated as feature. The index of all
        rows in X must be present in y.

        :param X: data sample with the features, which will be classified as relevant or not
        :type X: pandas.DataFrame or numpy.array

        :param y: target vector to be used, to classify the features
        :type y: pandas.Series or numpy.array

        :return: the fitted estimator with the information, which features are relevant
        :rtype: FeatureSelector
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X.copy())

        if not isinstance(y, pd.Series):
            y = pd.Series(y.copy())

        relevance_table = calculate_relevance_table(
            X,
            y,
            ml_task=self.ml_task,
            multiclass=self.multiclass,
            n_significant=self.n_significant,
            n_jobs=self.n_jobs,
            chunksize=self.chunksize,
            fdr_level=self.fdr_level,
            hypotheses_independent=self.hypotheses_independent,
            test_for_binary_target_real_feature=self.test_for_binary_target_real_feature,
        )
        self.relevant_features = relevance_table.loc[
            relevance_table.relevant
        ].feature.tolist()

        if self.multiclass:
            p_values_table = relevance_table.filter(regex="^p_value_*", axis=1)
            if self.multiclass_p_values == "all":
                self.p_values = p_values_table
                self.feature_importances_ = 1.0 - p_values_table
                self.feature_importances_.columns = self.feature_importances_.columns.str.lstrip(
                    "p_value"
                )
                self.feature_importances_ = self.feature_importances_.add_prefix(
                    "importance_"
                )
            elif self.multiclass_p_values == "min":
                self.p_values = p_values_table.min(axis=1).values
            elif self.multiclass_p_values == "max":
                self.p_values = p_values_table.max(axis=1).values
            elif self.multiclass_p_values == "avg":
                self.p_values = p_values_table.mean(axis=1).values

            if self.multiclass_p_values != "all":
                # raise p_values to the power of n_significant to increase importance
                # of features which are significant for more classes
                self.feature_importances_ = (
                    1.0 - self.p_values ** relevance_table.n_significant.values
                )
        else:
            self.feature_importances_ = 1.0 - relevance_table.p_value.values
            self.p_values = relevance_table.p_value.values

        self.features = relevance_table.index.tolist()

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
