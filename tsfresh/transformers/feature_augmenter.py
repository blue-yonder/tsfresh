# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import restrict_input_to_index


class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible estimator, for calculating and adding many features calculated from a given time series
    to the data. Is is basically a wrapper around :func:`~tsfresh.feature_extraction.extract_features`.

    The features include basic ones like min, max or median, and advanced features like fourier
    transformations or statistical tests. For a list of all possible features, see the module
    :mod:`~tsfresh.feature_extraction.feature_calculators`. The column name of each added feature contains the name
    of the function of that module, which was used for the calculation.

    For this estimator, two datasets play a crucial role:

    1. the time series container with the timeseries data. This container (for the format see :ref:`data-formats-label`)
       contains the data which is used for calculating the
       features. It must be groupable by ids which are used to identify which feature should be attached to which row
       in the second dataframe:

    2. the input data, where the features will be added to.

    Imagine the following situation: You want to classify 10 different financial shares and you have their development
    in the last year as a time series. You would then start by creating features from the metainformation of the
    shares, e.g. how long they were on the market etc. and filling up a table - the features of one stock in one row.

    >>> df = pandas.DataFrame()
    >>> # Fill in the information of the stocks
    >>> df["started_since_days"] = 0 # add a feature

    You can then extract all the features from the time development of the shares, by using this estimator:

    >>> time_series = read_in_timeseries() # get the development of the shares
    >>> from tsfresh.transformers import FeatureAugmenter
    >>> augmenter = FeatureAugmenter()
    >>> augmenter.set_timeseries_container(time_series)
    >>> df_with_time_series_features = augmenter.transform(df)

    The settings for the feature calculation can be controlled with the settings object. If you pass ``None``, the default
    settings are used. Please refer to :class:`~tsfresh.feature_extraction.settings.FeatureExtractionSettings` for
    more information.

    This estimator does not select the relevant features, but calculates and adds all of them to the DataFrame. See the
    :class:`~tsfresh.transformers.relevant_feature_augmenter.RelevantFeatureAugmenter` for calculating and selecting
    features.

    For a description what the parameters column_id, column_sort, column_kind and column_value mean, please see
    :mod:`~tsfresh.feature_extraction.extraction`.
    """
    def __init__(self, settings=None, column_id=None, column_sort=None,
                 column_kind=None, column_value=None, timeseries_container=None):
        """
        Create a new FeatureAugmenter instance.

        :param settings: The extraction settings to use. Leave empty to use the default ones.
        :type settings: tsfresh.feature_extraction.settings.FeatureExtractionSettings
        :param column_id: The column with the id. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_id: basestring
        :param column_sort: The column with the sort data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_sort: basestring
        :param column_kind: The column with the kind data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_kind: basestring
        :param column_value: The column with the values. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_value: basestring
        """
        self.settings = settings
        self.column_id = column_id
        self.column_sort = column_sort
        self.column_kind = column_kind
        self.column_value = column_value

        self.timeseries_container = timeseries_container

    def set_timeseries_container(self, timeseries_container):
        """
        Set the timeseries, with which the features will be calculated. For a format of the time series container,
        please refer to :mod:`~tsfresh.feature_extraction.extraction`. The timeseries must contain the same indices
        as the later DataFrame, to which the features will be added (the one you will pass to :func:`~transform`). You
        can call this function as often as you like, to change the timeseries later (e.g. if you want to extract for
        different ids).

        :param timeseries_container: The timeseries as a pandas.DataFrame or a dict. See
            :mod:`~tsfresh.feature_extraction.extraction` for the format.
        :type timeseries_container: pandas.DataFrame or dict
        :return: None
        :rtype: None
        """
        self.timeseries_container = timeseries_container

    def fit(self, X=None, y=None):
        """
        The fit function is not needed for this estimator. It just does nothing and is here for compatibility reasons.

        :param X: Unneeded.
        :type X: Any

        :param y: Unneeded.
        :type y: Any

        :return: The estimator instance itself
        :rtype: FeatureAugmenter
        """
        return self

    def transform(self, X):
        """
        Add the features calculated using the timeseries_container and add them to the corresponding rows in the input
        pandas.DataFrame X.

        To save some computing time, you should only include those time serieses in the container, that you
        need. You can set the timeseries container with the method :func:`set_timeseries_container`.

        :param X: the DataFrame to which the calculated timeseries features will be added. This is *not* the
               dataframe with the timeseries itself.
        :type X: pandas.DataFrame

        :return: The input DataFrame, but with added features.
        :rtype: pandas.DataFrame
        """
        if self.timeseries_container is None:
            raise RuntimeError("You have to provide a time series using the set_timeseries_container function before.")

        # Extract only features for the IDs in X.index
        timeseries_container_X = restrict_input_to_index(self.timeseries_container, self.column_id, X.index)

        extracted_features = extract_features(timeseries_container_X,
                                              feature_extraction_settings=self.settings,
                                              column_id=self.column_id, column_sort=self.column_sort,
                                              column_kind=self.column_kind, column_value=self.column_value)

        X = pd.merge(X, extracted_features, left_index=True, right_index=True, how="left")

        return X
