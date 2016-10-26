# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
from tsfresh.feature_extraction.settings import FeatureExtractionSettings
from tsfresh.transformers.feature_augmenter import FeatureAugmenter
from tsfresh.transformers.feature_selector import FeatureSelector
from tsfresh.utilities.dataframe_functions import impute_dataframe_range, get_range_values_per_column


# TODO: Add more testcases
# TODO: Do we want to keep the flag `evaluate_only_added_features`?
# Pro: It offers more control
# Contra: The Transformer is more than an Augmenter
class RelevantFeatureAugmenter(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible estimator to calculate relevant features out of a time series and add them to a data sample.

    As many other sklearn estimators, this estimator works in two steps:

    In the fit phase, all possible time series features are calculated using the time series, that is set by the
    set_timeseries_container function (if the features are not manually changed by handing in a
    feature_extraction_settings object). Then, their significance and relevance to the target is computed using
    statistical methods and only the relevant ones are selected using the Benjamini Hochberg procedure. These features
    are stored internally.

    In the transform step, the information on which features are relevant from the fit step is used and those features
    are extracted from the time series. These extracted features are then added to the input data sample.

    This estimator is a wrapper around most of the functionality in the tsfresh package. For more information on the
    subtasks, please refer to the single modules and functions, which are:

    * Settings for the feature extraction: :class:`~tsfresh.feature_extraction.settings.FeatureExtractionSettings`
    * Feature extraction method: :func:`~tsfresh.feature_extraction.extraction.extract_features`
    * Extracted features: :mod:`~tsfresh.feature_extraction.feature_calculators`
    * Settings for the feature selection: :class:`~tsfresh.feature_selection.settings.FeatureSignificanceTestsSettings`
    * Feature selection: :func:`~tsfresh.feature_selection.feature_selector.check_fs_sig_bh`

    This estimator works quite analogues to the :class:`~tsfresh.transformers.feature_augmenter.FeatureAugmenter` with
    the difference that this estimator does only output and calculate the relevant features,
    whereas the other outputs all features.

    Also for this estimator, two datasets play a crucial role:

    1. the time series container with the timeseries data. This container (for the format see
       :mod:`~tsfresh.feature_extraction.extraction`) contains the data which is used for calculating the
       features. It must be groupable by ids which are used to identify which feature should be attached to which row
       in the second dataframe:

    2. the input data, where the features will be added to.

    Imagine the following situation: You want to classify 10 different financial shares and you have their development
    in the last year as a time series. You would then start by creating features from the metainformation of the
    shares, e.g. how long they were on the market etc. and filling up a table - the features of one stock in one row.

    >>> # Fill in the information of the stocks and the target
    >>> X_train, X_test, y_train = pd.DataFrame(), pd.DataFrame(), pd.Series()

    You can then extract all the relevant features from the time development of the shares, by using this estimator:

    >>> train_time_series, test_time_series = read_in_timeseries() # get the development of the shares
    >>> from tsfresh.transformers import RelevantFeatureAugmenter
    >>> augmenter = RelevantFeatureAugmenter()
    >>> augmenter.set_timeseries_container(train_time_series)
    >>> augmenter.fit(X_train, y_train)
    >>> augmenter.set_timeseries_container(test_time_series)
    >>> X_test_with_features = augmenter.transform(X_test)

    X_test_with_features will then contain the same information as X_test (with all the meta information you have
    probably added) plus some relevant time series features calculated on the time series you handed in.

    Please keep in mind that the time series you hand in before fit or transform must contain data for the rows that are
    present in X.

    If your set evaluate_only_added_features to True, your manually-created features that were present in X_train (or
    X_test) before using this estimator are not touched. Otherwise, also those features are evaluated and may be
    rejected from the data sample, because they are irrelevant.

    For a description what the parameters column_id, column_sort, column_kind and column_value mean, please see
    :mod:`~tsfresh.feature_extraction.extraction`.

    You can control the feature extraction in the fit step (the feature extraction in the transform step is done
    automatically) as well as the feature selection in the fit step by handing in settings objects of the type
    :class:`~tsfresh.feature_extraction.settings.FeatureExtractionSettings` and
    :class:`~tsfresh.feature_selection.settings.FeatureSignificanceTestsSettings`. However, the default settings which
    are used if you pass no objects are often quite sensible.
    """
    def __init__(self,
                 evaluate_only_added_features=True,
                 feature_selection_settings=None,
                 feature_extraction_settings=None,
                 column_id=None, column_sort=None, column_kind=None, column_value=None,
                 timeseries_container=None):

        """
        Create a new RelevantFeatureAugmenter instance.

        :param settings: The extraction settings to use. Leave empty to use the default ones.
        :type settings: tsfresh.feature_extraction.settings.FeatureExtractionSettings

        :param evaluate_only_added_features: Whether to touch the manually-created features during feature selection or
                                             not.
        :type evaluate_only_added_features: bool
        :param feature_selection_settings: The feature selection settings.
        :type feature_selection_settings: tsfresh.feature_selection.settings.FeatureSelectionSettings
        :param feature_extraction_settings: The feature extraction settings.
        :type feature_selection_settings: tsfresh.feature_extraction.settings.FeatureExtractionSettings
        :param column_id: The column with the id. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_id: basestring
        :param column_sort: The column with the sort data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_sort: basestring
        :param column_kind: The column with the kind data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_kind: basestring
        :param column_value: The column with the values. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_value: basestring
        """

        # We require to have IMPUTE!
        if feature_extraction_settings is None:
            feature_extraction_settings = FeatureExtractionSettings()

        # Range will be our default imputation strategy
        feature_extraction_settings.IMPUTE = impute_dataframe_range

        self.feature_extractor = FeatureAugmenter(feature_extraction_settings,
                                                  column_id, column_sort, column_kind, column_value)

        self.feature_selector = FeatureSelector(feature_selection_settings)

        self.evaluate_only_added_features = evaluate_only_added_features

        self.timeseries_container = timeseries_container

    def set_timeseries_container(self, timeseries_container):
        """
        Set the timeseries, with which the features will be calculated. For a format of the time series container,
        please refer to :mod:`~tsfresh.feature_extraction.extraction`. The timeseries must contain the same indices
        as the later DataFrame, to which the features will be added (the one you will pass to :func:`~transform` or
        :func:`~fit`). You can call this function as often as you like, to change the timeseries later
        (e.g. if you want to extract for different ids).

        :param timeseries_container: The timeseries as a pandas.DataFrame or a dict. See
            :mod:`~tsfresh.feature_extraction.extraction` for the format.
        :type timeseries_container: pandas.DataFrame or dict
        :return: None
        :rtype: None
        """
        self.timeseries_container = timeseries_container

    def fit(self, X, y):
        """
        Use the given timeseries from :func:`~set_timeseries_container` and calculate features from it and add them
        to the data sample X (which can contain other manually-designed features).

        Then determine which of the features of X are relevant for the given target y.
        Store those relevant features internally to only extract them in the transform step.

        If evaluate_only_added_features is True, only reject newly, automatically added features. If it is False,
        also look at the features that are already present in the DataFrame.

        :param X: The data frame without the time series features. The index rows should be present in the timeseries
           and in the target vector.
        :type X: pandas.DataFrame or numpy.array

        :param y: The target vector to define, which features are relevant.
        :type y: pandas.Series or numpy.array

        :return: the fitted estimator with the information, which features are relevant.
        :rtype: RelevantFeatureAugmenter
        """
        if self.timeseries_container is None:
            raise RuntimeError("You have to provide a time series using the set_timeseries_container function before.")

        self.feature_extractor.set_timeseries_container(self.timeseries_container)

        if self.evaluate_only_added_features:
            # Do not merge the time series features to the old features
            X_tmp = pd.DataFrame(index=X.index)
        else:
            X_tmp = X

        X_augmented = self.feature_extractor.transform(X_tmp)

        if self.feature_extractor.settings.IMPUTE is impute_dataframe_range:
            self.col_to_max, self.col_to_min, self.col_to_median = get_range_values_per_column(X_augmented)

        self.feature_selector.fit(X_augmented, y)

        return self

    def transform(self, X):
        """
        After the fit step, it is known which features are relevant, Only extract those from the time series handed in
        with the function :func:`~set_timeseries_container`.

        If evaluate_only_added_features is False, also delete the irrelevant, already present features in the data frame.

        :param X: the data sample to add the relevant (and delete the irrelevant) features to.
        :type X: pandas.DataFrame or numpy.array

        :return: a data sample with the same information as X, but with added relevant time series features and
            deleted irrelevant information (only if evaluate_only_added_features is False).
        :rtype: pandas.DataFrame
        """
        if self.feature_selector.relevant_features is None:
            raise RuntimeError("You have to call fit before.")

        if self.timeseries_container is None:
            raise RuntimeError("You have to provide a time series using the set_timeseries_container function before.")

        self.feature_extractor.set_timeseries_container(self.timeseries_container)

        relevant_time_series_features = set(self.feature_selector.relevant_features) - set(pd.DataFrame(X).columns)

        relevant_extraction_settings = FeatureExtractionSettings.from_columns(relevant_time_series_features)
        relevant_extraction_settings.set_default = False

        # Set imputing strategy
        if self.feature_extractor.settings.IMPUTE is impute_dataframe_range:
            relevant_extraction_settings.IMPUTE = partial(impute_dataframe_range, col_to_max=self.col_to_max,
                                                          col_to_min=self.col_to_min, col_to_median=self.col_to_median)
        else:
            relevant_extraction_settings.IMPUTE = self.feature_extractor.settings.IMPUTE

        relevant_feature_extractor = FeatureAugmenter(settings=relevant_extraction_settings,
                                                      column_id=self.feature_extractor.column_id,
                                                      column_sort=self.feature_extractor.column_sort,
                                                      column_kind=self.feature_extractor.column_kind,
                                                      column_value=self.feature_extractor.column_value)

        relevant_feature_extractor.set_timeseries_container(self.feature_extractor.timeseries_container)

        X_augmented = relevant_feature_extractor.transform(X)

        return X_augmented.copy().loc[:, self.feature_selector.relevant_features]
