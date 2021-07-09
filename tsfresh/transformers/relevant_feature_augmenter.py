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
from tsfresh.utilities.dataframe_functions import (
    get_range_values_per_column,
    impute_dataframe_range,
)


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

    def __init__(
        self,
        filter_only_tsfresh_features=True,
        default_fc_parameters=None,
        kind_to_fc_parameters=None,
        column_id=None,
        column_sort=None,
        column_kind=None,
        column_value=None,
        timeseries_container=None,
        chunksize=defaults.CHUNKSIZE,
        n_jobs=defaults.N_PROCESSES,
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
        hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT,
        ml_task="auto",
        multiclass=False,
        n_significant=1,
        multiclass_p_values="min",
    ):
        """
        Create a new RelevantFeatureAugmenter instance.


        :param filter_only_tsfresh_features: Whether to touch the manually-created features during feature selection or
                                             not.
        :type filter_only_tsfresh_features: bool

        :param default_fc_parameters: mapping from feature calculator names to parameters. Only those names
               which are keys in this dict will be calculated. See the class:`ComprehensiveFCParameters` for
               more information.
        :type default_fc_parameters: dict

        :param kind_to_fc_parameters: mapping from kind names to objects of the same type as the ones for
                default_fc_parameters. If you put a kind as a key here, the fc_parameters
                object (which is the value), will be used instead of the default_fc_parameters. This means that kinds,
                for which kind_of_fc_parameters doe not have any entries, will be ignored by the feature selection.
        :type kind_to_fc_parameters: dict
        :param column_id: The column with the id. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_id: basestring
        :param column_sort: The column with the sort data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_sort: basestring
        :param column_kind: The column with the kind data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_kind: basestring
        :param column_value: The column with the values. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_value: basestring

        :param chunksize: The size of one chunk that is submitted to the worker
            process for the parallelisation.  Where one chunk is defined as a
            singular time series for one id and one kind. If you set the chunksize
            to 10, then it means that one task is to calculate all features for 10
            time series.  If it is set it to None, depending on distributor,
            heuristics are used to find the optimal chunksize. If you get out of
            memory exceptions, you can try it with the dask distributor and a
            smaller chunksize.
        :type chunksize: None or int

        :param n_jobs: The number of processes to use for parallelization. If zero, no parallelization is used.
        :type n_jobs: int

        :param show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
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
        self.filter_only_tsfresh_features = filter_only_tsfresh_features
        self.default_fc_parameters = default_fc_parameters
        self.kind_to_fc_parameters = kind_to_fc_parameters
        self.column_id = column_id
        self.column_sort = column_sort
        self.column_kind = column_kind
        self.column_value = column_value
        self.timeseries_container = timeseries_container
        self.chunksize = chunksize
        self.n_jobs = n_jobs
        self.show_warnings = show_warnings
        self.disable_progressbar = disable_progressbar
        self.profile = profile
        self.profiling_filename = profiling_filename
        self.profiling_sorting = profiling_sorting
        self.test_for_binary_target_binary_feature = (
            test_for_binary_target_binary_feature
        )
        self.test_for_binary_target_real_feature = test_for_binary_target_real_feature
        self.test_for_real_target_binary_feature = test_for_real_target_binary_feature
        self.test_for_real_target_real_feature = test_for_real_target_real_feature
        self.fdr_level = fdr_level
        self.hypotheses_independent = hypotheses_independent
        self.ml_task = ml_task
        self.multiclass = multiclass
        self.n_significant = n_significant
        self.multiclass_p_values = multiclass_p_values

        # attributes
        self.feature_extractor = None
        self.feature_selector = None

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
        self._fit_and_augment(X, y)
        return self

    def transform(self, X):
        """
        After the fit step, it is known which features are relevant, Only extract those from the time series handed in
        with the function :func:`~set_timeseries_container`.

        If filter_only_tsfresh_features is False, also delete the irrelevant,
        already present features in the data frame.

        :param X: the data sample to add the relevant (and delete the irrelevant) features to.
        :type X: pandas.DataFrame or numpy.array

        :return: a data sample with the same information as X, but with added relevant time series features and
            deleted irrelevant information (only if filter_only_tsfresh_features is False).
        :rtype: pandas.DataFrame
        """

        if self.timeseries_container is None:
            raise RuntimeError(
                "You have to provide a time series using the set_timeseries_container function before."
            )

        if self.feature_selector is None:
            raise RuntimeError("You have to call fit before calling transform.")

        if self.feature_selector.relevant_features is None:
            raise RuntimeError("You have to call fit before calling transform.")

        self.feature_extractor.set_timeseries_container(self.timeseries_container)

        relevant_time_series_features = set(
            self.feature_selector.relevant_features
        ) - set(pd.DataFrame(X).columns)
        relevant_extraction_settings = from_columns(relevant_time_series_features)

        # Set imputing strategy
        impute_function = partial(
            impute_dataframe_range,
            col_to_max=self.col_to_max,
            col_to_min=self.col_to_min,
            col_to_median=self.col_to_median,
        )

        relevant_feature_extractor = FeatureAugmenter(
            kind_to_fc_parameters=relevant_extraction_settings,
            default_fc_parameters={},
            column_id=self.feature_extractor.column_id,
            column_sort=self.feature_extractor.column_sort,
            column_kind=self.feature_extractor.column_kind,
            column_value=self.feature_extractor.column_value,
            chunksize=self.feature_extractor.chunksize,
            n_jobs=self.feature_extractor.n_jobs,
            show_warnings=self.feature_extractor.show_warnings,
            disable_progressbar=self.feature_extractor.disable_progressbar,
            impute_function=impute_function,
            profile=self.feature_extractor.profile,
            profiling_filename=self.feature_extractor.profiling_filename,
            profiling_sorting=self.feature_extractor.profiling_sorting,
        )

        relevant_feature_extractor.set_timeseries_container(
            self.feature_extractor.timeseries_container
        )

        X_augmented = relevant_feature_extractor.transform(X)

        if self.filter_only_tsfresh_features:
            return X_augmented.copy().loc[
                :, self.feature_selector.relevant_features + X.columns.tolist()
            ]
        else:
            return X_augmented.copy().loc[:, self.feature_selector.relevant_features]

    def fit_transform(self, X, y):
        """
        Equivalent to :func:`~fit` followed by :func:`~transform`; however, this is faster than performing those steps
        separately, because it avoids re-extracting relevant features for training data.

        :param X: The data frame without the time series features. The index rows should be present in the timeseries
           and in the target vector.
        :type X: pandas.DataFrame or numpy.array

        :param y: The target vector to define, which features are relevant.
        :type y: pandas.Series or numpy.array

        :return: a data sample with the same information as X, but with added relevant time series features and
            deleted irrelevant information (only if filter_only_tsfresh_features is False).
        :rtype: pandas.DataFrame
        """
        X_augmented = self._fit_and_augment(X, y)

        selected_features = X_augmented.copy().loc[
            :, self.feature_selector.relevant_features
        ]

        if self.filter_only_tsfresh_features:
            selected_features = pd.merge(
                selected_features, X, left_index=True, right_index=True, how="left"
            )

        return selected_features

    def _fit_and_augment(self, X, y):
        """
        Helper for the :func:`~fit` and :func:`~fit_transform` functions, which does most of the work described in
        :func:`~fit`.

        :param X: The data frame without the time series features. The index rows should be present in the timeseries
           and in the target vector.
        :type X: pandas.DataFrame or numpy.array

        :param y: The target vector to define, which features are relevant.
        :type y: pandas.Series or numpy.array

        :return: a data sample with the extraced time series features. If filter_only_tsfresh_features is False
            the data sample will also include the information in X.
        :rtype: pandas.DataFrame
        """
        if self.timeseries_container is None:
            raise RuntimeError(
                "You have to provide a time series using the set_timeseries_container function before."
            )

        self.feature_extractor = FeatureAugmenter(
            default_fc_parameters=self.default_fc_parameters,
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            column_id=self.column_id,
            column_sort=self.column_sort,
            column_kind=self.column_kind,
            column_value=self.column_value,
            timeseries_container=self.timeseries_container,
            chunksize=self.chunksize,
            n_jobs=self.n_jobs,
            show_warnings=self.show_warnings,
            disable_progressbar=self.disable_progressbar,
            profile=self.profile,
            profiling_filename=self.profiling_filename,
            profiling_sorting=self.profiling_sorting,
        )

        self.feature_selector = FeatureSelector(
            test_for_binary_target_binary_feature=self.test_for_binary_target_binary_feature,
            test_for_binary_target_real_feature=self.test_for_binary_target_real_feature,
            test_for_real_target_binary_feature=self.test_for_real_target_binary_feature,
            test_for_real_target_real_feature=self.test_for_real_target_real_feature,
            fdr_level=self.fdr_level,
            hypotheses_independent=self.hypotheses_independent,
            n_jobs=self.n_jobs,
            chunksize=self.chunksize,
            ml_task=self.ml_task,
            multiclass=self.multiclass,
            n_significant=self.n_significant,
            multiclass_p_values=self.multiclass_p_values,
        )

        if self.filter_only_tsfresh_features:
            # Do not merge the time series features to the old features
            X_tmp = pd.DataFrame(index=X.index)
        else:
            X_tmp = X

        X_augmented = self.feature_extractor.transform(X_tmp)

        (
            self.col_to_max,
            self.col_to_min,
            self.col_to_median,
        ) = get_range_values_per_column(X_augmented)
        X_augmented = impute_dataframe_range(
            X_augmented,
            col_to_max=self.col_to_max,
            col_to_median=self.col_to_median,
            col_to_min=self.col_to_min,
        )

        self.feature_selector.fit(X_augmented, y)

        return X_augmented
