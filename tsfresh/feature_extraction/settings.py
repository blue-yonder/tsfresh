# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This file contains methods/objects for controlling which features will be extracted when calling extract_features.
For the naming of the features, see :ref:`feature-naming-label`.
"""

from __future__ import absolute_import, division

from inspect import getargspec

import numpy as np
from builtins import range
from past.builtins import basestring

from itertools import product

from tsfresh.feature_extraction import feature_calculators
from tsfresh.utilities.string_manipulation import get_config_from_string


def from_columns(columns, columns_to_ignore=[]):
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
    :param columns_to_ignore: columns which do not contain tsfresh feature names
    :type columns_to_ignore: list of str

    :return: The kind_to_fc_parameters object ready to be used in the extract_features function.
    :rtype: dict
    """

    kind_to_fc_parameters = {}

    for col in columns:
        if col in columns_to_ignore:
            continue

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

        config = get_config_from_string(parts)
        if config:
            if feature_name in kind_to_fc_parameters[kind]:
                kind_to_fc_parameters[kind][feature_name].append(config)
            else:
                kind_to_fc_parameters[kind][feature_name] = [config]
        else:
            kind_to_fc_parameters[kind][feature_name] = None

    return kind_to_fc_parameters


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
            if callable(func) and hasattr(func, "fctype") and len(getargspec(func).args) == 1:
                name_to_param[name] = None

        name_to_param.update({
            "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 4)],
            "c3": [{"lag": lag} for lag in range(1, 4)],
            "cid_ce": [{"normalize": True}, {"normalize": False}],
            "symmetry_looking": [{"r": r * 0.05} for r in range(20)],
            "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
            "quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]],
            "autocorrelation": [{"lag": lag} for lag in range(10)],
            "agg_autocorrelation": [{"f_agg": s} for s in ["mean", "median", "var"]],
            "partial_autocorrelation": [{"lag": lag} for lag in range(10)],
            "number_cwt_peaks": [{"n": n} for n in [1, 5]],
            "number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
            "binned_entropy": [{"max_bins": max_bins} for max_bins in [10]],
            "index_mass_quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]],
            "cwt_coefficients": [{"widths": width, "coeff": coeff, "w": w} for
                                 width in [(2, 5, 10, 20)] for coeff in range(15) for w in (2, 5, 10, 20)],
            "spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]],
            "ar_coefficient": [{"coeff": coeff, "k": k} for coeff in range(5) for k in [10]],
            "change_quantiles": [{"ql": ql, "qh": qh, "isabs": b, "f_agg": f}
                                          for ql in [0., .2, .4, .6, .8] for qh in [.2, .4, .6, .8, 1.]
                                          for b in [False, True] for f in ["mean", "var"]],
            "fft_coefficient": [{"coeff": k, "attr": a} for a, k in product(["real", "imag", "abs", "angle"], range(100))],
            "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]],
            "value_count": [{"value": value} for value in [0, 1, np.NaN, np.PINF, np.NINF]],
            "range_count": [{"min": -1, "max": 1}],
            "approximate_entropy": [{"m": 2, "r": r} for r in [.1, .3, .5, .7, .9]],
            "friedrich_coefficients": (lambda m: [{"coeff": coeff, "m": m, "r": 30} for coeff in range(m + 1)])(3),
            "max_langevin_fixed_point": [{"m": 3, "r": 30}],
            "linear_trend": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"},
                             {"attr": "slope"}, {"attr": "stderr"}],
            "agg_linear_trend": [{"attr": attr, "chunk_len": i, "f_agg": f}
                                 for attr in ["rvalue", "intercept", "slope", "stderr"]
                                 for i in [5, 10, 50]
                                 for f in ["max", "min", "mean", "var"]],
            "augmented_dickey_fuller": [{"attr": "teststat"}, {"attr": "pvalue"}, {"attr": "usedlag"}],
            "number_crossing_m": [{"m": 0}, {"m": -1}, {"m": 1}],
            "energy_ratio_by_chunks": [{"num_segments" : 10, "segment_focus": i} for i in range(10)],
            "ratio_beyond_r_sigma": [{"r": x} for x in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]]
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
