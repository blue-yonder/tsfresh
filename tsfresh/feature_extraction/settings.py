# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This file contains methods/objects for controlling which features will be extracted when calling extract_features.
For the naming of the features, see :ref:`feature-naming-label`.
"""

from __future__ import absolute_import, division

import ast
from functools import partial

import numpy as np
from builtins import range
from builtins import str
from builtins import zip
from past.builtins import basestring
from tsfresh.feature_extraction import feature_calculators


def get_aggregate_functions(fc_parameters, column_prefix):
    """
    Returns a dictionary with some of the column name mapped to the feature calculators that are
    specified in the fc_parameters. This dictionary includes those calculators,
    that can be used in a pandas group by command to extract all aggregate features at the same time.

    :param fc_parameters: mapping from feature calculator names to settings.
    :type fc_parameters: ComprehensiveFCParameters or child class
    :param column_prefix: the prefix for all column names.
    :type column_prefix: basestring

    :return: mapping of column name to function calculator
    :rtype: dict
    """

    aggregate_functions = {}

    for name, param in fc_parameters.items():

        func = getattr(feature_calculators, name)

        if func.fctype == "aggregate":

            aggregate_functions['{}__{}'.format(column_prefix, name)] = func

        elif func.fctype == "aggregate_with_parameters":

            if not isinstance(param, list):
                raise ValueError("The parameters needs to be saved as a list of dictionaries")

            for config in param:

                if not isinstance(config, dict):
                    raise ValueError("The parameters needs to be saved as a list of dictionaries")

                # if there are several params, create a feature for each one
                c = '{}__{}'.format(column_prefix, name)
                for arg, p in sorted(config.items()):
                    c += "__" + arg + "_" + str(p)
                aggregate_functions[c] = partial(func, **config)

        elif func.fctype == "apply":
            pass
        else:
            raise ValueError("Do not know fctype {}".format(func.fctype))

    return aggregate_functions


def get_apply_functions(fc_parameters, column_prefix):
    """
    Returns a dictionary with some of the column name mapped to the feature calculators that are
    specified in the fc_parameters. This dictionary includes those calculators,
    that can *not* be used in a pandas group by command to extract all aggregate features at the same time.

    :param fc_parameters: mapping from feature calculator names to settings.
    :type fc_parameters: ComprehensiveFCParameters or child class

    :param column_prefix: the prefix for all column names.
    :type column_prefix: basestring

    :return: all functions to use for feature extraction
    :rtype: list
    """

    apply_functions = []

    for name, param in fc_parameters.items():

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


def from_columns(columns):
    """
    Creates a mapping from kind names to fc_parameters objects
    (which are itself mappings from feature calculators to settings)
    to extract only the features contained in the columns.
    To do so, for every feature name in columns this method

    1. split the column name into col, feature, params part
    2. decide which feature we are dealing with (aggregate with/without params or apply)
    3. add it to the new name_to_function dict
    4. set up the params

    :param columns: containing the feature names
    :type columns: list of str

    :return: The kind_to_fc_parameters object ready to be used in the extract_features function.
    :rtype: dict
    """

    kind_to_fc_parameters = {}

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

        if kind not in kind_to_fc_parameters:
            kind_to_fc_parameters[kind] = {}

        if not hasattr(feature_calculators, feature_name):
            raise ValueError("Unknown feature name {}".format(feature_name))

        func = getattr(feature_calculators, feature_name)

        if func.fctype == "aggregate":

            kind_to_fc_parameters[kind][feature_name] = None

        elif func.fctype == "aggregate_with_parameters":

            config = _get_config_from_string(parts)

            if feature_name in kind_to_fc_parameters[kind]:
                kind_to_fc_parameters[kind][feature_name].append(config)
            else:
                kind_to_fc_parameters[kind][feature_name] = [config]

        elif func.fctype == "apply":

            config = _get_config_from_string(parts)

            if feature_name in kind_to_fc_parameters[kind]:
                kind_to_fc_parameters[kind][feature_name].append(config)
            else:
                kind_to_fc_parameters[kind][feature_name] = [config]

    return kind_to_fc_parameters


def _get_config_from_string(parts):
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


# todo: this classes' docstrings are not completely up-to-date
class ComprehensiveFCParameters(dict):
    """
    Create a new ComprehensiveFCParameters instance. You have to pass this instance to the
    extract_feature instance.

    It is basically a dictionary (and also based on one), which is a mapping from
    string (the same names that are in the feature_calculators.py file) to a list of dictionary of parameters,
    which should be used when the function with this name is called.

    Only those strings (function names), that are keys in this dictionary, will be later used to extract
    features - so whenever you delete a key from this dict, you disable the calculation of this feature.

    You can use the settings object with

    >>> from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters
    >>> extract_features(df, default_fc_parameters=ComprehensiveFCParameters())

    to extract all features (which is the default nevertheless) or you change the ComprehensiveFCParameters
    object to other types (see below).
    """
    def __init__(self):
        name_to_param = {}

        for name, func in feature_calculators.__dict__.items():
            if callable(func):
                if hasattr(func, "fctype") and getattr(func, "fctype") == "aggregate":
                    name_to_param[name] = None

        name_to_param.update({
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
            "approximate_entropy": [{"m": 2, "r": r} for r in [.1, .3, .5, .7, .9]],
            "friedrich_coefficients": (lambda m: [{"coeff": coeff, "m": m, "r": 30} for coeff in range(m + 1)])(3),
            "max_langevin_fixed_point": [{"m": 3, "r": 30}],
        })

        super(ComprehensiveFCParameters, self).__init__(name_to_param)


class MinimalFCParameters(ComprehensiveFCParameters):
    """
    This class is a child class of the ComprehensiveFCParameters class
    and has the same functionality as its base class. The only difference is,
    that most of the feature calculators are disabled and only a small
    subset of calculators will be calculated at all. Those are donated by an attribute called "minimal".

    Use this class for quick tests of your setup before calculating all
    features which could take some time depending of your data set size.

    You should use this object when calling the extract function, like so:

    >>> from tsfresh.feature_extraction import extract_features, MinimalFCParameters
    >>> extract_features(df, default_fc_parameters=MinimalFCParameters())
    """

    def __init__(self):
        ComprehensiveFCParameters.__init__(self)

        for fname, f in feature_calculators.__dict__.items():
            if fname in self and (not hasattr(f, "minimal") or not getattr(f, "minimal")):
                del self[fname]


class EfficientFCParameters(ComprehensiveFCParameters):
    """
    This class is a child class of the ComprehensiveFCParameters class
    and has the same functionality as its base class.

    The only difference is, that the features with high computational costs are not calculated. Those are denoted by
    the attribute "high_comp_cost".

    You should use this object when calling the extract function, like so:

    >>> from tsfresh.feature_extraction import extract_features, EfficientFCParameters
    >>> extract_features(df, default_fc_parameters=EfficientFCParameters())
    """

    def __init__(self):
        ComprehensiveFCParameters.__init__(self)

        # drop all features with high computational costs
        for fname, f in feature_calculators.__dict__.items():
            if hasattr(f, "high_comp_cost"):
                del self[fname]
