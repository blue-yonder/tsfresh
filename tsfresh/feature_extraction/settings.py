# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This file contains all settings of the tsfresh.
For the naming of the features, see :ref:`feature-naming-label`.

"""

from __future__ import absolute_import, division
from builtins import zip
from builtins import str
from builtins import range
from past.builtins import basestring
from builtins import object
import ast
import os
from functools import partial
import numpy as np
from tsfresh.feature_extraction import feature_calculators
from multiprocessing import cpu_count
import six


# todo: this classes' docstrings are not completely up-to-date
class FeatureExtractionSettings(object):
    """
    This class defines the behaviour of feature extraction, in particular which feature and parameter combinations are calculated.
    If you do not specify any user settings, all features will be extracted with default arguments defined in this class.

    In general, we consider three types of time series features:

    1. aggregate features without parameter that emit exactly one feature per function calculator
    2. aggregate features with parameter that emit exactly one feature per function calculator
    3. apply features with parameters that emit several features per function calculator (usually one feature per parameter value)

    These three types are stored in different dictionaries. For the feature types with parameters there is also a
    dictionaries containing the parameters.

    It is possible to obtain a `FeatureExtractionSettings` object from a feature matrix,
    see func:`~tsfresh.feature_extraction.settings.FeatureExtractionSettings.from_columns`. This is useful to reproduce
    the features of a train set for a test set.

    To set user defined settings, do something like

    >>> from tsfresh.feature_extraction import FeatureExtractionSettings
    >>> settings = FeatureExtractionSettings()
    >>> # Calculate all features except length
    >>> settings.do_not_calculate("length")
    >>> from tsfresh.feature_extraction import extract_features
    >>> extract_features(df, feature_extraction_settings=settings)

    Mostly, the settings in this class are for enabling/disabling the extraction of certain features, which can be
    important to save time during feature extraction. Additionally, some of the features have parameters which can be
    controlled here.

    If the calculation of a feature failed (for whatever reason), the results can be NaN. The IMPUTE flag defaults to
    `None` and can be set to one of the impute functions in :mod:`~tsfresh.utilities.dataframe_functions`.
    """

    def __init__(self, calculate_all_features=True):
        """
        Create a new FeatureExtractionSettings instance. You have to pass this instance to the
        extract_feature instance.
        """

        self.kind_to_calculation_settings_mapping = {}
        self.PROFILING = False
        self.PROFILING_SORTING = "cumulative"
        self.PROFILING_FILENAME = "profile.txt"
        self.IMPUTE = None
        self.set_default = True
        self.name_to_param = {}
        # Do not show the progress bar
        self.disable_progressbar = False

        # Set to false to dismiss all warnings.
        self.show_warnings = False

        if calculate_all_features is True:
            for name, func in feature_calculators.__dict__.items():
                if callable(func):
                    if hasattr(func, "fctype") and getattr(func, "fctype") == "aggregate":
                        self.name_to_param[name] = None
            self.name_to_param.update({
                "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 4)],
                "symmetry_looking": [{"r": r * 0.05} for r in range(20)],
                "large_standard_deviation": [{"r": r * 0.05} for r in range(10)],
                "quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]],
                "autocorrelation": [{"lag": lag} for lag in range(10)],
                "number_cwt_peaks": [{"n": n} for n in [1, 5]],
                "number_peaks": [{"n": n} for n in [1, 3, 5]],
                "large_number_of_peaks": [{"n": n} for n in [1, 3, 5]],
                "binned_entropy": [{"max_bins": max_bins} for max_bins in [10]],
                "index_mass_quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]],
                "cwt_coefficients": [{"widths": width, "coeff": coeff, "w": w} for
                                     width in [(2, 5, 10, 20)] for coeff in range(15) for w in (2, 5, 10, 20)],
                "spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]],
                "ar_coefficient": [{"coeff": coeff, "k": k} for coeff in range(5) for k in [10]],
                "mean_abs_change_quantiles": [{"ql": ql, "qh": qh}
                                              for ql in [0., .2, .4, .6, .8] for qh in [.2, .4, .6, .8, 1.]],
                "fft_coefficient": [{"coeff": coeff} for coeff in range(0, 10)],
                "value_count": [{"value": value} for value in [0, 1, np.NaN, np.PINF, np.NINF]],
                "range_count": [{"min": -1, "max": 1}],
                "approximate_entropy": [{"m": 2, "r": r} for r in [.1, .3, .5, .7, .9]]
            })

        # default None means one procesqs per cpu
        n_cores = int(os.getenv("NUMBER_OF_CPUS") or cpu_count())
        self.n_processes = max(1, n_cores//2)

        # Size of the chunks submitted to the worker processes
        self.chunksize = None

    def set_default_parameters(self, kind):
        """
        Setup the feature calculations for kind as defined in `self.name_to_param`

        :param kind: str, the type of the time series
        :return:
        """

        self.kind_to_calculation_settings_mapping[kind] = self.name_to_param.copy()

    def do_not_calculate(self, kind, identifier):
        """
        Delete the all features of type identifier for time series of type kind.

        :param kind: the type of the time series
        :type kind: basestring
        :param identifier: the name of the feature
        :type identifier: basestring
        :return: The setting object itself
        :rtype: FeatureExtractionSettings
        """

        if not isinstance(kind, basestring):
            raise TypeError("Time series {} should be a string".format(kind))
        if not isinstance(identifier, basestring):
            raise TypeError("Identifier {} should be a string".format(identifier))

        del self.kind_to_calculation_settings_mapping[kind][identifier]
        return self

    @staticmethod
    def from_columns(columns):
        """
        Creates a FeatureExtractionSettings object set to extract only the features contained in the list columns. to
        do so, for every feature name in columns this method

        1. split the column name into col, feature, params part
        2. decide which feature we are dealing with (aggregate with/without params or apply)
        3. add it to the new name_to_function dict
        4. set up the params

        Set the feature and params dictionaries in the settings object, then return it.

        :param columns: containing the feature names
        :type columns: list of str
        :return: The changed settings object
        :rtype: FeatureExtractionSettings
        """

        kind_to_calculation_settings_mapping = {}

        for col in columns:

            if not isinstance(col, basestring):
                raise TypeError("Column name {} should be a string or unicode".format(col))

            # Split according to our separator into <col_name>, <feature_name>, <feature_params>
            parts = col.split('__')
            n_parts = len(parts)

            if n_parts == 1:
                raise ValueError("Splitting of columnname {} resulted in only one part.".format(col))

            kind = parts[0]
            feature_name = parts[1]

            if kind not in kind_to_calculation_settings_mapping:
                kind_to_calculation_settings_mapping[kind] = {}

            if not hasattr(feature_calculators, feature_name):
                raise ValueError("Unknown feature name {}".format(feature_name))

            func = getattr(feature_calculators, feature_name)

            if func.fctype == "aggregate":

                kind_to_calculation_settings_mapping[kind][feature_name] = None

            elif func.fctype == "aggregate_with_parameters":

                config = FeatureExtractionSettings.get_config_from_string(parts)

                if feature_name in kind_to_calculation_settings_mapping[kind]:
                    kind_to_calculation_settings_mapping[kind][feature_name].append(config)
                else:
                    kind_to_calculation_settings_mapping[kind][feature_name] = [config]

            elif func.fctype == "apply":

                config = FeatureExtractionSettings.get_config_from_string(parts)

                if feature_name in kind_to_calculation_settings_mapping[kind]:
                    kind_to_calculation_settings_mapping[kind][feature_name].append(config)
                else:
                    kind_to_calculation_settings_mapping[kind][feature_name] = [config]

        settings = FeatureExtractionSettings()
        settings.kind_to_calculation_settings_mapping = kind_to_calculation_settings_mapping

        return settings

    @staticmethod
    def get_config_from_string(parts):
        """
        Helper function to extract the configuration of a certain function from the column name.
        The column name parts (split by "__") should be passed to this function. It will skip the
        kind name and the function name and only use the parameter parts. These parts will be split up on "_"
        into the parameter name and the parameter value. This value is transformed into a python object
        (for example is "(1, 2, 3)" transformed into a tuple consisting of the ints 1, 2 and 3).

        :param parts: The column name split up on "__"
        :type parts: list
        :return: a dictionary with all parameters, which are encoded in the column name.
        :rtype: dict
        """
        relevant_parts = parts[2:]
        config_kwargs = [s.rsplit("_", 1)[0] for s in relevant_parts]
        config_values = [s.rsplit("_", 1)[1] for s in relevant_parts]

        dict_if_configs = {}

        for key, value in zip(config_kwargs, config_values):
            if value.lower() == "nan":
                dict_if_configs[key] = np.NaN
            elif value.lower() == "-inf":
                dict_if_configs[key] = np.NINF
            elif value.lower() == "inf":
                dict_if_configs[key] = np.PINF
            else:
                dict_if_configs[key] = ast.literal_eval(value)

        return dict_if_configs

    def get_aggregate_functions(self, kind):
        """
        For the tyme series Returns a dictionary with the column name mapped to the feature calculators that are
        specified in the FeatureExtractionSettings object. This dictionary can be used in a pandas group by command to
        extract the all aggregate features at the same time.

        :param kind: the type of the time series
        :type kind: basestring
        :return: mapping of column name to function calculator
        :rtype: dict
        """

        aggregate_functions = {}

        if kind not in self.kind_to_calculation_settings_mapping:
            return aggregate_functions

        for name, param in self.kind_to_calculation_settings_mapping[kind].items():

            func = getattr(feature_calculators, name)

            if func.fctype == "aggregate":

                aggregate_functions['{}__{}'.format(kind, name)] = func

            elif func.fctype == "aggregate_with_parameters":

                if not isinstance(param, list):
                    raise ValueError("The parameters needs to be saved as a list of dictionaries")

                for config in param:

                    if not isinstance(config, dict):
                        raise ValueError("The parameters needs to be saved as a list of dictionaries")

                    # if there are several params, create a feature for each one
                    c = '{}__{}'.format(kind, name)
                    for arg, p in sorted(config.items()):
                        c += "__" + arg + "_" + str(p)
                    aggregate_functions[c] = partial(func, **config)

            elif func.fctype == "apply":
                pass
            else:
                raise ValueError("Do not know fctype {}".format(func.fctype))

        return aggregate_functions

    def get_apply_functions(self, column_prefix):
        """
        Convenience function to return a list with all the functions to apply on a data frame and extract features.
        Only adds those functions to the dictionary, that are enabled in the settings.

        :param column_prefix: the prefix all column names.
        :type column_prefix: basestring
        :return: all functions to use for feature extraction
        :rtype: list
        """

        apply_functions = []

        if column_prefix not in self.kind_to_calculation_settings_mapping:
            return apply_functions

        for name, param in self.kind_to_calculation_settings_mapping[column_prefix].items():

            func = getattr(feature_calculators, name)

            if func.fctype == "apply":

                if not isinstance(param, list):
                    raise ValueError("The parameters needs to be saved as a list of dictionaries")

                apply_functions.append((func, {"c": column_prefix, "param": param}))

            elif func.fctype == "aggregate" or func.fctype == "aggregate_with_parameters":
                pass
            else:
                raise ValueError("Do not know fctype {}".format(func.fctype))

        return apply_functions


class MinimalFeatureExtractionSettings(FeatureExtractionSettings):
    """
    This class is a child class of the FeatureExtractionSettings class
    and has the same functionality as its base class. The only difference is,
    that most of the feature calculators are disabled and only a small
    subset of calculators will be calculated at all.

    Use this class for quick tests of your setup before calculating all
    features which could take some time depending of your data set size.

    You should use this object when calling the extract function, like so:

    >>> from tsfresh.feature_extraction import extract_features, MinimalFeatureExtractionSettings
    >>> extract_features(df, feature_extraction_settings=MinimalFeatureExtractionSettings)
    """
    def __init__(self):
        FeatureExtractionSettings.__init__(self, True)

        name_to_param_copy = {}

        for feature_calculator in self.name_to_param:
            function = feature_calculators.__dict__[feature_calculator]

            if hasattr(function, "minimal") and getattr(function, "minimal"):
                name_to_param_copy[feature_calculator] = self.name_to_param[feature_calculator]

        self.name_to_param = name_to_param_copy


class ReasonableFeatureExtractionSettings(FeatureExtractionSettings):
    """
    This class is a child class of the FeatureExtractionSettings class
    and has the same functionality as its base class.

    The only difference is, that the features with high computational costs are not calculated. Those are denoted by
    the attribute "high_comp_cost"

    You should use this object when calling the extract function, like so:

    >>> from tsfresh.feature_extraction import extract_features, ReasonableFeatureExtractionSettings
    >>> extract_features(df, feature_extraction_settings=ReasonableFeatureExtractionSettings)
    """

    def __init__(self):
        FeatureExtractionSettings.__init__(self, True)

        # drop all features with high computational costs
        for fname, f in six.iteritems(feature_calculators.__dict__):
            if hasattr(f, "high_comp_cost"):
                del self.name_to_param[fname]