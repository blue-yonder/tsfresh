# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
from functools import partial

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh import defaults
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.transformers.feature_augmenter import FeatureAugmenter
from tsfresh.transformers.feature_selector import FeatureSelector
from tsfresh.utilities.dataframe_functions import impute_dataframe_range, get_range_values_per_column


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

    * Settings for the feature extraction: :class:`~tsfresh.feature_extraction.settings.ComprehensiveFCParameters`
    * Feature extraction method: :func:`~tsfresh.feature_extraction.extraction.extract_features`
    * Extracted features: :mod:`~tsfresh.feature_extraction.feature_calculators`
    * Feature selection: :func:`~tsfresh.feature_selection.feature_selector.check_fs_sig_bh`

    This estimator works analogue to the :class:`~tsfresh.transformers.feature_augmenter.FeatureAugmenter` with
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

    If your set filter_only_tsfresh_features to True, your manually-created features that were present in X_train (or
    X_test) before using this estimator are not touched. Otherwise, also those features are evaluated and may be
    rejected from the data sample, because they are irrelevant.

    For a description what the parameters column_id, column_sort, column_kind and column_value mean, please see
    :mod:`~tsfresh.feature_extraction.extraction`.

    You can control the feature extraction in the fit step (the feature extraction in the transform step is done
    automatically) as well as the feature selection in the fit step by handing in settings.
    However, the default settings which are used if you pass no flags are often quite sensible.
    """

    def __init__(self,
                 filter_only_tsfresh_features=True,
                 default_fc_parameters=None,
                 kind_to_fc_parameters=None,
                 column_id=None, column_sort=None, column_kind=None, column_value=None,
                 timeseries_container=None,
                 parallelization=defaults.PARALLELISATION,
                 chunksize=defaults.CHUNKSIZE,
                 impute_function=defaults.IMPUTE_FUNCTION,
                 n_processes=defaults.N_PROCESSES,
                 show_warnings=defaults.SHOW_WARNINGS,
                 disable_progressbar=defaults.DISABLE_PROGRESSBAR,
                 profile=defaults.PROFILING,
                 profiling_filename=defaults.PROFILING_FILENAME,
                 profiling_sorting=defaults.PROFILING_SORTING,
                 test_for_binary_target_binary_feature=defaults.TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
                 test_for_binary_target_real_feature=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE,
                 test_for_real_target_binary_feature=defaults.TEST_FOR_REAL_TARGET_BINARY_FEATURE,
                 test_for_real_target_real_feature=defaults.TEST_FOR_REAL_TARGET_REAL_FEATURE,
                 fdr_level=defaults.FDR_LEVEL,
                 hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT):

        """
        Create a new RelevantFeatureAugmenter instance.

        :param settings: The extraction settings to use. Leave empty to use the default ones.
        :type settings: tsfresh.feature_extraction.settings.ExtendedFCParameters

        :param filter_only_tsfresh_features: Whether to touch the manually-created features during feature selection or
                                             not.
        :type filter_only_tsfresh_features: bool
        :param feature_selection_settings: The feature selection settings.
        :type feature_selection_settings: tsfresh.feature_selection.settings.FeatureSelectionSettings
        :param feature_extraction_settings: The feature extraction settings.
        :type feature_selection_settings: tsfresh.feature_extraction.settings.ComprehensiveFCParameters
        :param column_id: The column with the id. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_id: basestring
        :param column_sort: The column with the sort data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_sort: basestring
        :param column_kind: The column with the kind data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_kind: basestring
        :param column_value: The column with the values. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_value: basestring

        :param parallelization: Either ``'per_sample'`` or ``'per_kind'``   , see
                            :func:`~tsfresh.feature_extraction.extraction._extract_features_parallel_per_sample`,
                            :func:`~tsfresh.feature_extraction.extraction._extract_features_parallel_per_kind` and
                            :ref:`parallelization-label` for details.
                            Choosing None makes the algorithm look for the best parallelization technique by applying
                            some general remarks.
        :type parallelization: str

        :param chunksize: The size of one chunk for the parallelisation
        :type chunksize: None or int

        :param n_processes: The number of processes to use for parallelisation.
        :type n_processes: int

        :param: show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
        :type show_warnings: bool

        :param disable_progressbar: Do not show a progressbar while doing the calculation.
        :type disable_progressbar: bool

        :param profile: Turn on profiling during feature extraction
        :type profile: bool

        :param profiling_sorting: How to sort the profiling results (see the documentation of the profiling package for
               more information)
        :type profiling_sorting: basestring

        :param profiling_filename: Where to save the profiling results.
        :type profiling_filename: basestring

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
        """

        self.feature_extractor = FeatureAugmenter(column_id=column_id, column_sort=column_sort, column_kind=column_kind,
                                                  column_value=column_value,
                                                  default_fc_parameters=default_fc_parameters,
                                                  kind_to_fc_parameters=kind_to_fc_parameters,
                                                  parallelization=parallelization, chunksize=chunksize,
                                                  n_processes=n_processes, show_warnings=show_warnings,
                                                  disable_progressbar=disable_progressbar,
                                                  impute_function=impute_function,
                                                  profile=profile,
                                                  profiling_filename=profiling_filename,
                                                  profiling_sorting=profiling_sorting
                                                  )

        self.feature_selector = FeatureSelector(
            test_for_binary_target_binary_feature=test_for_binary_target_binary_feature,
            test_for_binary_target_real_feature=test_for_binary_target_real_feature,
            test_for_real_target_binary_feature=test_for_real_target_binary_feature,
            test_for_real_target_real_feature=test_for_real_target_real_feature,
            fdr_level=fdr_level, hypotheses_independent=hypotheses_independent,
            n_processes=n_processes, chunksize=chunksize
        )

        self.filter_only_tsfresh_features = filter_only_tsfresh_features
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

        If filter_only_tsfresh_features is True, only reject newly, automatically added features. If it is False,
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

        if self.filter_only_tsfresh_features:
            # Do not merge the time series features to the old features
            X_tmp = pd.DataFrame(index=X.index)
        else:
            X_tmp = X

        X_augmented = self.feature_extractor.transform(X_tmp)

        if self.feature_extractor.impute_function is impute_dataframe_range:
            self.col_to_max, self.col_to_min, self.col_to_median = get_range_values_per_column(X_augmented)

        self.feature_selector.fit(X_augmented, y)

        return self

    def transform(self, X):
        """
        After the fit step, it is known which features are relevant, Only extract those from the time series handed in
        with the function :func:`~set_timeseries_container`.

        If filter_only_tsfresh_features is False, also delete the irrelevant, already present features in the data frame.

        :param X: the data sample to add the relevant (and delete the irrelevant) features to.
        :type X: pandas.DataFrame or numpy.array

        :return: a data sample with the same information as X, but with added relevant time series features and
            deleted irrelevant information (only if filter_only_tsfresh_features is False).
        :rtype: pandas.DataFrame
        """
        if self.feature_selector.relevant_features is None:
            raise RuntimeError("You have to call fit before.")

        if self.timeseries_container is None:
            raise RuntimeError("You have to provide a time series using the set_timeseries_container function before.")

        self.feature_extractor.set_timeseries_container(self.timeseries_container)

        relevant_time_series_features = set(self.feature_selector.relevant_features) - set(pd.DataFrame(X).columns)
        relevant_extraction_settings = from_columns(relevant_time_series_features)

        # Set imputing strategy
        if self.feature_extractor.impute_function is impute_dataframe_range:
            self.feature_extractor.impute_function = partial(impute_dataframe_range,
                                                             col_to_max=self.col_to_max,
                                                             col_to_min=self.col_to_min,
                                                             col_to_median=self.col_to_median)

        relevant_feature_extractor = FeatureAugmenter(kind_to_fc_parameters=relevant_extraction_settings,
                                                      default_fc_parameters={},
                                                      column_id=self.feature_extractor.column_id,
                                                      column_sort=self.feature_extractor.column_sort,
                                                      column_kind=self.feature_extractor.column_kind,
                                                      column_value=self.feature_extractor.column_value,
                                                      parallelization=self.feature_extractor.parallelization,
                                                      chunksize=self.feature_extractor.chunksize,
                                                      n_processes=self.feature_extractor.n_processes,
                                                      show_warnings=self.feature_extractor.show_warnings,
                                                      disable_progressbar=self.feature_extractor.disable_progressbar,
                                                      impute_function=self.feature_extractor.impute_function,
                                                      profile=self.feature_extractor.profile,
                                                      profiling_filename=self.feature_extractor.profiling_filename,
                                                      profiling_sorting=self.feature_extractor.profiling_sorting)

        relevant_feature_extractor.set_timeseries_container(self.feature_extractor.timeseries_container)

        X_augmented = relevant_feature_extractor.transform(X)

        if self.filter_only_tsfresh_features:
            return X_augmented.copy().loc[:, self.feature_selector.relevant_features + X.columns.tolist()]
        else:
            return X_augmented.copy().loc[:, self.feature_selector.relevant_features]
