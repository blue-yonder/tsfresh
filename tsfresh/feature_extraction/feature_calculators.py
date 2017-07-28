# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This module contains the feature calculators that take time series as input and calculate the values of the feature.
There are two types of features:

1. feature calculators which calculate a single number (simple)
2. feature calculators which calculate a bunch of features for a list of parameters at once,
   to use e.g. cached results (combiner). They return a list of (key, value) pairs for each input parameter.

They are specified using the "fctype" parameter of each feature calculator, which is added using the
set_property function. Only functions in this python module, which have a parameter called  "fctype" are
seen by tsfresh as a feature calculator. Others will not be calculated.
"""

from __future__ import absolute_import, division
from builtins import range
import itertools
import numpy as np
from numpy.linalg import LinAlgError

import pandas as pd
from scipy.signal import welch, cwt, ricker, find_peaks_cwt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from scipy.stats import linregress

# todo: make sure '_' works in parameter names in all cases, add a warning if not


def _get_length_sequences_where(x):
    """
    This method calculates the length of all sub-sequences where the array x is either True or 1.

    Examples
    --------
    >>> x = [0,1,0,0,1,1,1,0,0,1,0,1,1]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,True,True,True,0,0,True,0,True,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,1,True,1,0,0,True,0,1,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    :param x: An iterable containing only 1, True, 0 and False values
    :return: A list with the length of all sub-sequences where the array is either True or False. If no ones or Trues
    contained, the list [0] is returned.
    """
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]


def _estimate_friedrich_coefficients(x, m, r):
    """
    Coefficients of polynomial :math:`h(x)`, which has been fitted to
    the deterministic dynamics of Langevin model
    .. math::
        \dot{x}(t) = h(x(t)) + \mathcal{N}(0,R)

    As described by

        Friedrich et al. (2000): Physics Letters A 271, p. 217-222
        *Extracting model equations from experimental data*

    For short time-series this method is highly dependent on the parameters.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param m: order of polynom to fit for estimating fixed points of dynamics
    :type m: int
    :param r: number of quantils to use for averaging
    :type r: float

    :return: coefficients of polynomial of deterministic dynamics
    :return type: ndarray
    """
    df = pd.DataFrame({'signal': x[:-1], 'delta': np.diff(x)})
    try:
        df['quantiles'] = pd.qcut(df.signal, r)
    except ValueError:
        return [np.NaN] * (m + 1)

    quantiles = df.groupby('quantiles')

    result = pd.DataFrame({'x_mean': quantiles.signal.mean(),
                           'y_mean': quantiles.delta.mean()
                           })

    result.dropna(inplace=True)

    try:
        return np.polyfit(result.x_mean, result.y_mean, deg=m)
    except (np.linalg.LinAlgError, ValueError):
        return [np.NaN] * (m + 1)


def _aggregate_on_chunks(x, f_agg, chunk_len):
    """
    Takes the time series x and constructs a lower sampled version of it by applying the aggregation function f_agg on
    consecutive chunks of length chunk_len

    :param x: the time series to calculate the aggregation of
    :type x: pandas.Series
    :param f_agg: The name of the aggregation function that should be an attribute of the pandas.Series
    :type f_agg: str
    :param chunk_len: The size of the chunks where to aggregate the time series
    :type chunk_len: int
    :return: A list of the aggregation function over the chunks
    :return type: list
    """
    return [getattr(x[i * chunk_len: (i + 1) * chunk_len], f_agg)() for i in range(int(np.ceil(len(x) / chunk_len)))]


def set_property(key, value):
    """
    This method returns a decorator that sets the property key of the function to value
    """
    def decorate_func(func):
        setattr(func, key, value)
        if func.__doc__ and key == "fctype":
            func.__doc__ = func.__doc__ + "\n\n    *This function is of type: " + value + "*\n"
        return func
    return decorate_func


@set_property("fctype", "simple")
def variance_larger_than_standard_deviation(x):
    """
    Boolean variable denoting if the variance of x is greater than its standard deviation. Is equal to variance of x
    being larger than 1

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return np.var(x) > np.std(x)


@set_property("fctype", "simple")
def ratio_beyond_r_sigma(x, r):
    """
    Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.

    :param x: the time series to calculate the feature of
    :type x: iterable
    :return: the value of this feature
    :return type: bool
    """
    x = np.asarray(x)
    return sum(abs(x - np.mean(x)) > r * np.std(x))/len(x)


@set_property("fctype", "simple")
def large_standard_deviation(x, r):
    """
    Boolean variable denoting if the standard dev of x is higher
    than 'r' times the range = difference between max and min of x.
    Hence it checks if

    .. math::

        std(x) > r * (max(X)-min(X))

    According to a rule of the thumb, the standard deviation should be a forth of the range of the values.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param r: the percentage of the range to compare with
    :type r: float
    :return: the value of this feature
    :return type: bool
    """
    x = np.asarray(x)
    return np.std(x) > (r * (max(x) - min(x)))


@set_property("fctype", "combiner")
def symmetry_looking(x, param):
    """
    Boolean variable denoting if the distribution of x *looks symmetric*. This is the case if

    .. math::

        | mean(X)-median(X)| < r * (max(X)-min(X))

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param r: the percentage of the range to compare with
    :type r: float
    :return: the value of this feature
    :return type: bool
    """
    x = np.asarray(x)
    mean_median_difference = abs(np.mean(x) - np.median(x))
    max_min_difference = max(x) - min(x)
    return [("r_{}".format(r["r"]), mean_median_difference < (r["r"] * max_min_difference))
            for r in param]


@set_property("fctype", "simple")
def has_duplicate_max(x):
    """
    Checks if the maximum value of x is observed more than once

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return sum(np.asarray(x) == max(x)) >= 2


@set_property("fctype", "simple")
def has_duplicate_min(x):
    """
    Checks if the minimal value of x is observed more than once

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return sum(np.asarray(x) == min(x)) >= 2


@set_property("fctype", "simple")
def has_duplicate(x):
    """
    Checks if any value in x occurs more than once

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return len(x) != len(set(x))


@set_property("fctype", "simple")
@set_property("minimal", True)
def sum_values(x):
    """
    Calculates the sum over the time series values

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return np.sum(x)


@set_property("fctype", "combiner")
def agg_autocorrelation(x, param):
    """
    Calculates the value of an aggregation function f_agg (e.g. var or mean) of the autocorrelation
    (Compare to http://en.wikipedia.org/wiki/Autocorrelation#Estimation), taken over different all possible lags
    (1 to length of x)

    .. math::

        \\frac{1}{n-1} \\sum_{l=1,\ldots, n} \\frac{1}{(n-l)\sigma^{2}} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)

    where :math:`n` is the length of the time series :math:`X_i`, :math:`\sigma^2` its variance and :math:`\mu` its
    mean.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"attr": x} with x str, name of a numpy function (e.g. mean, var, std, median),
                   the name of the aggregator function that is applied to the autocorrelations
    :type param: list
    :return: the value of this feature
    :return type: float
    """
    var = np.var(x)
    n = len(x)
    if abs(var) < 10**-10 or n == 1:
        a = 0
    else:
        a = acf(x, unbiased=True, fft=n > 1250)[1:]
    return [("f_agg_\"{}\"".format(config["f_agg"]), getattr(np, config["f_agg"])(a)) for config in param]


@set_property("fctype", "combiner")
def augmented_dickey_fuller(x, param):
    """
    The Augmented Dickey-Fuller test is a hypothesis test which checks whether a unit root is present in a time
    series sample. This feature calculator returns the value of the respective test statistic.

    See the statsmodels implementation for references and more details.
    
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"attr": x} with x str, either "teststat", "pvalue" or "usedlag"
    :type param: list
    :return: the value of this feature
    :return type: float
    """
    res = None
    try:
        res = adfuller(x)
    except LinAlgError:
        res = np.NaN, np.NaN, np.NaN
    except ValueError:  # occurs if sample size is too small
        res = np.NaN, np.NaN, np.NaN

    return [("attr_{}".format(config["attr"]),
                  res[0] if config["attr"] == "teststat"
             else res[1] if config["attr"] == "pvalue"
             else res[2] if config["attr"] == "usedlag" else np.NaN)
            for config in param]


@set_property("fctype", "simple")
def abs_energy(x):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\ldots, n} x_i^2

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return sum(x * x)


@set_property("fctype", "simple")
def mean_abs_change(x):
    """
    Returns the mean over the absolute differences between subsequent time series values which is

    .. math::

        \\frac{1}{n} \\sum_{i=1,\ldots, n-1} | x_{i+1} - x_{i}|


    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.mean(abs(np.diff(x)))


@set_property("fctype", "simple")
def mean_change(x):
    """
    Returns the mean over the absolute differences between subsequent time series values which is

    .. math::

        \\frac{1}{n} \\sum_{i=1,\ldots, n-1}  x_{i+1} - x_{i}

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.mean(np.diff(x))


@set_property("fctype", "simple")
def mean_second_derivate_central(x):
    """
    Returns the mean value of a central approximation of the second derivative

    .. math::

        \\frac{1}{n} \\sum_{i=1,\ldots, n-1}  \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_i)

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """

    diff = (np.roll(x, 1) - 2 * np.array(x) + np.roll(x, -1)) / 2.0
    return np.mean(diff[1:-1])


@set_property("fctype", "simple")
@set_property("minimal", True)
def median(x):
    """
    Returns the median of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.median(x)


@set_property("fctype", "simple")
@set_property("minimal", True)
def mean(x):
    """
    Returns the mean of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.mean(x)


@set_property("fctype", "simple")
@set_property("minimal", True)
def length(x):
    """
    Returns the length of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: int
    """
    return len(x)


@set_property("fctype", "simple")
@set_property("minimal", True)
def standard_deviation(x):
    """
    Returns the standard deviation of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.std(x)


@set_property("fctype", "simple")
@set_property("minimal", True)
def variance(x):
    """
    Returns the variance of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.var(x)


@set_property("fctype", "simple")
def skewness(x):
    """
    Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1).

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = pd.Series(x)
    return pd.Series.skew(x)


@set_property("fctype", "simple")
def kurtosis(x):
    """
    Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G2).

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = pd.Series(x)
    return pd.Series.kurtosis(x)


@set_property("fctype", "simple")
def absolute_sum_of_changes(x):
    """
    Returns the sum over the absolute value of consecutive changes in the series x

    .. math::

        \\sum_{i=1, \ldots, n-1} \\mid x_{i+1}- x_i \\mid

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.sum(abs(np.diff(x)))


@set_property("fctype", "simple")
def longest_strike_below_mean(x):
    """
    Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """

    return max(_get_length_sequences_where(x <= np.mean(x))) if len(x) > 0 else 0


@set_property("fctype", "simple")
def longest_strike_above_mean(x):
    """
    Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """

    return max(_get_length_sequences_where(x >= np.mean(x))) if len(x) > 0 else 0


@set_property("fctype", "simple")
def count_above_mean(x):
    """
    Returns the number of values in x that are higher than the mean of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """

    x = np.asarray(x)
    m = np.mean(x)
    return np.where(x > m)[0].shape[0]


@set_property("fctype", "simple")
def count_below_mean(x):
    """
    Returns the number of values in x that are lower than the mean of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """

    x = np.asarray(x)
    m = np.mean(x)
    return np.where(x < m)[0].shape[0]


@set_property("fctype", "simple")
def last_location_of_maximum(x):
    """
    Returns the relative last location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN


@set_property("fctype", "simple")
def first_location_of_maximum(x):
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN


@set_property("fctype", "simple")
def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN


@set_property("fctype", "simple")
def first_location_of_minimum(x):
    """
    Returns the first location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return np.argmin(x) / len(x) if len(x) > 0 else np.NaN


@set_property("fctype", "simple")
def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    """
    Returns the percentage of unique values, that are present in the time series
    more than once.

        len(different values occurring more than once) / len(different values)

    This means the percentage is normalized to the number of unique values,
    in contrast to the percentage_of_reoccurring_values_to_all_values.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """

    unique, counts = np.unique(x, return_counts=True)
    return np.sum(counts > 1) / float(counts.shape[0])


@set_property("fctype", "simple")
def percentage_of_reoccurring_values_to_all_values(x):
    """
    Returns the ratio of unique values, that are present in the time series
    more than once.

        # of data points occurring more than once / # of all data points

    This means the ratio is normalized to the number of data points in the time series,
    in contrast to the percentage_of_reoccurring_datapoints_to_all_datapoints.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = pd.Series(x)

    if len(x) == 0:
        return np.nan

    value_counts = x.value_counts()
    return value_counts[value_counts > 1].sum() / len(x)


@set_property("fctype", "simple")
def sum_of_reoccurring_values(x):
    """
    Returns the sum of all values, that are present in the time series
    more than once.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    counts[counts > 1] = 1
    return np.sum(counts * unique)


@set_property("fctype", "simple")
def sum_of_reoccurring_data_points(x):
    """
    Returns the sum of all data points, that are present in the time series
    more than once.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    return np.sum(counts * unique)


@set_property("fctype", "simple")
def ratio_value_number_to_time_series_length(x):
    """
    Returns a factor which is 1 if all values in the time series occur only once,
    and below one if this is not the case.
    In principle, it just returns

        # unique values / # values

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """

    if len(x) == 0:
        return np.nan

    return len(set(x))/len(x)


@set_property("fctype", "combiner")
def fft_coefficient(x, param):
    """
    Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast
    fourier transformation algorithm

    .. math::
        A_k =  \\sum_{m=0}^{n-1} a_m \\exp \\left \\{ -2 \\pi i \\frac{m k}{n} \\right \\}, \\qquad k = 0, \\ldots , n-1.

    The resulting coefficients will be complex, this feature calculator can return the real part (attr=="real"),
    the imaginary part (attr=="imag), the absolute value (attr=""abs) and the angle in degrees (attr=="angle).

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"coeff": x, "attr": s} with x int and x >= 0, s str and in ["real", "imag",
        "abs", "angle"]
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    assert min([config["coeff"] for config in param]) >= 0, "Coefficients must be positive or zero."
    assert set([config["attr"] for config in param]) <= set(["imag", "real", "abs", "angle"]), \
        'Attribute must be "real", "imag", "angle" or "abs"'

    fft = np.fft.rfft(x)

    def complex_agg(x, agg):
        if agg == "real":
            return x.real
        elif agg == "imag":
            return x.imag
        elif agg == "abs":
            return np.abs(x)
        elif agg == "angle":
            return np.angle(x, deg=True)

    res = [complex_agg(fft[config["coeff"]], config["attr"]) if config["coeff"] < len(fft)
           else np.NaN for config in param]
    index = ['coeff_{}__attr_"{}"'.format(config["coeff"], config["attr"]) for config in param]
    return zip(index, res)


@set_property("fctype", "simple")
def number_peaks(x, n):
    """
    Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
    subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.

    Hence in the sequence

    >>> x = [3, 0, 0, 4, 0, 0, 13]

    4 is a peak of support 1 and 2 because in the subsequences

    >>> [0, 4, 0]
    >>> [0, 0, 4, 0, 0]

    4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
    and its bigger than 4.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param n: the support of the peak
    :type n: int
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    x_reduced = x[n:-n]

    res = None
    for i in range(1, n + 1):
        result_first = (x_reduced > np.roll(x, i)[n:-n])

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= (x_reduced > np.roll(x, -i)[n:-n])
    return sum(res)


@set_property("fctype", "combiner")
def index_mass_quantile(x, param):
    """
    Those apply features calculate the relative index i where q% of the mass of the time series x lie left of i.
    For example for q = 50% this feature calculator will return the mass center of the time series

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"q": x} with x float
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    x = np.asarray(x)
    abs_x = np.abs(x)
    s = sum(abs_x)

    if s == 0:
        # all values in x are zero or it has length 0
        return [("q_{}".format(config["q"]), np.NaN) for config in param]
    else:
        # at least one value is not zero
        mass_centralized = np.cumsum(abs_x) / s
        return [("q_{}".format(config["q"]),
                (np.argmax(mass_centralized >= config["q"])+1)/len(x)) for config in param]


@set_property("fctype", "simple")
def number_cwt_peaks(x, n):
    """
    This feature calculator searches for different peaks in x. To do so, x is smoothed by a ricker wavelet and for
    widths ranging from 1 to n. This feature calculator returns the number of peaks that occur at enough width scales
    and with sufficiently high Signal-to-Noise-Ratio (SNR)

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param n: maximum width to consider
    :type n: int
    :return: the value of this feature
    :return type: int
    """
    return len(find_peaks_cwt(vector=x, widths=np.array(list(range(1, n + 1))), wavelet=ricker))


@set_property("fctype", "combiner")
def linear_trend(x, param):
    """
    Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to
    length of the time series minus one.
    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.
    The parameters control which of the characteristics are returned.

    Possible extracted attributes are "pvalue", "rvalue", "intercept", "slope", "stderr", see the documentation of
    linregress for more information.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"attr": x} with x an string, the attribute name of the regression model
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    # todo: we could use the index of the DataFrame here

    linReg = linregress(range(len(x)), x)

    return [("attr_\"{}\"".format(config["attr"]), getattr(linReg, config["attr"]))
            for config in param]


@set_property("fctype", "combiner")
def cwt_coefficients(x, param):
    """
    Calculates a Continuous wavelet transform for the Ricker wavelet, also known as the "Mexican hat wavelet" which is
    defined by

    .. math::
        \\frac{2}{\\sqrt{3a} \\pi^{\\frac{1}{4}}} (1 - \\frac{x^2}{a^2}) exp(-\\frac{x^2}{2a^2})

    where :math:`a` is the width parameter of the wavelet function.

    This feature calculator takes three different parameter: widths, coeff and w. The feature calculater takes all the
    different widths arrays and then calculates the cwt one time for each different width array. Then the values for the
    different coefficient for coeff and width w are returned. (For each dic in param one feature is returned)

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"widths":x, "coeff": y, "w": z} with x array of int and y,z int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    calculated_cwt = {}
    res = []
    indices = []

    for parameter_combination in param:
        widths = parameter_combination["widths"]
        w = parameter_combination["w"]
        coeff = parameter_combination["coeff"]

        if widths not in calculated_cwt:
            calculated_cwt[widths] = cwt(x, ricker, widths)

        calculated_cwt_for_widths = calculated_cwt[widths]

        indices += ["widths_{}__coeff_{}__w_{}".format(widths, coeff, w)]

        i = widths.index(w)
        if calculated_cwt_for_widths.shape[1] <= coeff:
            res += [np.NaN]
        else:
            res += [calculated_cwt_for_widths[i, coeff]]

    return zip(indices, res)


@set_property("fctype", "combiner")
def spkt_welch_density(x, param):
    """
    This feature calculator estimates the cross power spectral density of the time series x at different frequencies.
    To do so, the time series is first shifted from the time domain to the frequency domain.

    The feature calculators returns the power spectrum of the different frequencies.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"coeff": x} with x int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    freq, pxx = welch(x)
    coeff = [config["coeff"] for config in param]
    indices = ["coeff_{}".format(i) for i in coeff]

    if len(pxx) <= max(coeff):  # There are fewer data points in the time series than requested coefficients

        # filter coefficients that are not contained in pxx
        reduced_coeff = [coefficient for coefficient in coeff if len(pxx) > coefficient]
        not_calculated_coefficients = [coefficient for coefficient in coeff
                                       if coefficient not in reduced_coeff]

        # Fill up the rest of the requested coefficients with np.NaNs
        return zip(indices, list(pxx[reduced_coeff]) + [np.NaN] * len(not_calculated_coefficients))
    else:
        return zip(indices, pxx[coeff])


@set_property("fctype", "combiner")
def ar_coefficient(x, param):
    """
    This feature calculator fits the unconditional maximum likelihood
    of an autoregressive AR(k) process.
    The k parameter is the maximum lag of the process

    .. math::

        X_{t}=\\varphi_0 +\\sum _{{i=1}}^{k}\\varphi_{i}X_{{t-i}}+\\varepsilon_{t}

    For the configurations from param which should contain the maxlag "k" and such an AR process is calculated. Then
    the coefficients :math:`\\varphi_{i}` whose index :math:`i` contained from "coeff" are returned.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"coeff": x, "k": y} with x,y int
    :type param: list
    :return x: the different feature values
    :return type: pandas.Series
    """
    calculated_ar_params = {}

    x_as_list = list(x)
    calculated_AR = AR(x_as_list)

    res = {}

    for parameter_combination in param:
        k = parameter_combination["k"]
        p = parameter_combination["coeff"]

        column_name = "k_{}__coeff_{}".format(k, p)

        if k not in calculated_ar_params:
            try:
                calculated_ar_params[k] = calculated_AR.fit(maxlag=k, solver="mle").params
            except (LinAlgError, ValueError):
                calculated_ar_params[k] = [np.NaN]*k

        mod = calculated_ar_params[k]

        if p <= k:
            try:
                res[column_name] = mod[p]
            except IndexError:
                res[column_name] = 0
        else:
            res[column_name] = np.NaN

    return [(key, value) for key, value in res.items()]


@set_property("fctype", "simple")
def change_quantiles(x, ql, qh, isabs, f_agg):
    """
    First fixes a corridor given by the quantiles ql and qh of the distribution of x.
    Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.

    Think about selecting a corridor on the
    y-Axis and only calculating the mean of the absolute change of the time series inside this corridor.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param ql: the lower quantile of the corridor
    :type ql: float
    :param qh: the higher quantile of the corridor
    :type qh: float
    :param isabs: should the absolute differences be taken?
    :type isabs: bool
    :param f_agg: the aggregator function that is applied to the differences in the bin
    :type f_agg: str, name of a numpy function (e.g. mean, var, std, median)

    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)

    if ql >= qh:
        ValueError("ql={} should be lower than qh={}".format(ql, qh))

    div = np.diff(x)
    if isabs:
        div = np.abs(div)
    # All values that originate from the corridor between the quantiles ql and qh will have the category 0,
    # other will be np.NaN
    try:
        bin_cat = pd.qcut(x, [ql, qh], labels=False)
        bin_cat_0 = bin_cat == 0
    except ValueError:  # Occurs when ql are qh effectively equal, e.g. x is not long enough or is too categorical
        return 0
    # We only count changes that start and end inside the corridor
    ind = (bin_cat_0 * np.roll(bin_cat_0, 1))[1:]
    if sum(ind) == 0:
        return 0
    else:
        ind_inside_corridor = np.where(ind == 1)
        aggregator = getattr(np, f_agg)
        return aggregator(div[ind_inside_corridor])



@set_property("fctype", "simple")
def time_reversal_asymmetry_statistic(x, lag):
    """
    This function calculates the value of

    .. math::

        \\frac{1}{n-2lag} \sum_{i=0}^{n-2lag} x_{i + 2 \cdot lag}^2 \cdot x_{i + lag} - x_{i + lag} \cdot  x_{i}^2

    which is

    .. math::

        \\mathbb{E}[L^2(X)^2 \cdot L(X) - L(X) \cdot X^2]

    where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a
    promising feature to extract from time series.

    References
    ----------

    .. [1] Fulcher, B.D., Jones, N.S. (2014).
       Highly comparative feature-based time-series classification.
       Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.


    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param lag: the lag that should be used in the calculation of the feature
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    n = len(x)
    x = np.asarray(x)
    if 2 * lag >= n:
        return 0
    else:
        return np.mean((np.roll(x, 2 * -lag) * np.roll(x, 2 * -lag) * np.roll(x, -lag) -
                        np.roll(x, -lag) * x * x)[0:(n - 2 * lag)])


@set_property("fctype", "simple")
def c3(x, lag):
    """
    This function calculates the value of

    .. math::

        \\frac{1}{n-2lag} \sum_{i=0}^{n-2lag} x_{i + 2 \cdot lag}^2 \cdot x_{i + lag} \cdot x_{i}

    which is

    .. math::

        \\mathbb{E}[L^2(X)^2 \cdot L(X) \cdot X]

    where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a measure of
    non linearity in the time series.

    References
    ----------

    .. [1] Schreiber, T. and Schmitz, A. (1997).
       Discrimination power of measures for nonlinearity in a time series
       PHYSICAL REVIEW E, VOLUME 55, NUMBER 5

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param lag: the lag that should be used in the calculation of the feature
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    n = len(x)
    x = np.asarray(x)
    if 2 * lag >= n:
        return 0
    else:
        return np.mean((np.roll(x, 2 * -lag) * np.roll(x, -lag) * x)[0:(n - 2 * lag)])


@set_property("fctype", "simple")
def binned_entropy(x, max_bins):
    """
    First bins the values of x into max_bins equidistant bins.
    Then calculates the value of

    .. math::

        - \\sum_{k=0}^{min(max\\_bins, len(x))} p_k log(p_k) \\cdot \\mathbf{1}_{(p_k > 0)}

    where :math:`p_k` is the percentage of samples in bin :math:`k`.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param max_bins: the maximal number of bins
    :type max_bins: int
    :return: the value of this feature
    :return type: float
    """
    hist, bin_edges = np.histogram(x, bins=max_bins)
    probs = hist / len(x)
    return - np.sum(p * np.math.log(p) for p in probs if p != 0)

# todo - include latex formula
# todo - check if vectorizable
@set_property("high_comp_cost", True)
@set_property("fctype", "simple")
def sample_entropy(x):
    """
    Calculate and return sample entropy of x.
    References:
    ----------
    [1] http://en.wikipedia.org/wiki/Sample_Entropy
    [2] https://www.ncbi.nlm.nih.gov/pubmed/10843903?dopt=Abstract
    
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param tolerance: normalization factor; equivalent to the common practice of expressing the tolerance as r times the standard deviation
    :type tolerance: float
    :return: the value of this feature
    :return type: float
    """ 
    x = np.array(x)
    
    sample_length = 1 # number of sequential points of the time series
    tolerance = 0.2 * np.std(x) # 0.2 is a common value for r - why?

    n = len(x)
    prev = np.zeros(n)
    curr = np.zeros(n)
    A = np.zeros((1, 1))  # number of matches for m = [1,...,template_length - 1]
    B = np.zeros((1, 1))  # number of matches for m = [1,...,template_length]

    for i in range(n - 1):
        nj = n - i - 1
        ts1 = x[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - ts1) < tolerance:  # distance between two vectors
                curr[jj] = prev[jj] + 1
                temp_ts_length = min(sample_length, curr[jj])
                for m in range(int(temp_ts_length)):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
            else:
                curr[jj] = 0
        for j in range(nj):
            prev[j] = curr[j]
            
    N = n * (n - 1) / 2
    B = np.vstack(([N], B[0]))
    
    # sample entropy = -1 * (log (A/B))
    similarity_ratio = A / B 
    se = -1 * np.log(similarity_ratio)
    se = np.reshape(se, -1)
    return se[0]
    

@set_property("fctype", "simple")
def autocorrelation(x, lag):
    """
    Calculates the lag autocorrelation of a lag value of lag.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param lag: the lag
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    x = pd.Series(x)
    return pd.Series.autocorr(x, lag)


@set_property("fctype", "simple")
def quantile(x, q):
    """
    Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param q: the quantile to calculate
    :type q: float
    :return: the value of this feature
    :return type: float
    """
    x = pd.Series(x)
    return pd.Series.quantile(x, q)


@set_property("fctype", "simple")
def number_crossing_m(x, m):
    """
    Calculates the number of crossings of x on m. A crossing is defined as two sequential values where the first value
    is lower than m and the next is greater, or vice-versa. If you set m to zero, you will get the number of zero
    crossings.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param m: the threshold for the crossing
    :type m: float
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    x = x[x != m]
    return sum(np.abs(np.diff(np.sign(x - m))))/2


@set_property("fctype", "simple")
@set_property("minimal", True)
def maximum(x):
    """
    Calculates the highest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return max(x)


@set_property("fctype", "simple")
@set_property("minimal", True)
def minimum(x):
    """
    Calculates the lowest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return min(x)


@set_property("fctype", "simple")
def value_count(x, value):
    """
    Count occurrences of `value` in time series x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param value: the value to be counted
    :type value: int or float
    :return: the count
    :rtype: int
    """
    if np.isnan(value):
        return np.isnan(x).sum()
    else:
        return x[x == value].shape[0]


@set_property("fctype", "simple")
def range_count(x, min, max):
    """
    Count observed values within the interval [min, max).

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param min: the inclusive lower bound of the range
    :type min: int or float
    :param max: the exclusive upper bound of the range
    :type max: int or float
    :return: the count of values within the range
    :rtype: int
    """
    return np.sum((x >= min) & (x < max))


@set_property("fctype", "simple")
@set_property("high_comp_cost", True)
def approximate_entropy(x, m, r):
    """
    Implements a vectorized Approximate entropy algorithm.

        https://en.wikipedia.org/wiki/Approximate_entropy

    For short time-series this method is highly dependent on the parameters,
    but should be stable for N > 2000, see:

        Yentes et al. (2012) - 
        *The Appropriate Use of Approximate Entropy and Sample Entropy with Short Data Sets*


    Other shortcomings and alternatives discussed in:

        Richman & Moorman (2000) -
        *Physiological time-series analysis using approximate entropy and sample entropy*

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param m: Length of compared run of data
    :type m: int
    :param r: Filtering level, must be positive
    :type r: float

    :return: Approximate entropy
    :return type: float
    """
    x = np.asarray(x)
    N = len(x)
    r *= np.std(x)
    if r < 0:
        raise ValueError("Parameter r must be positive.")
    if N <= m+1:
        return 0

    def _phi(m):
        x_re = np.array([x[i:i+m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x_re[:, np.newaxis] - x_re[np.newaxis, :]),
                          axis=2) <= r, axis=0) / (N-m+1)
        return np.sum(np.log(C)) / (N - m + 1.0)

    return np.abs(_phi(m) - _phi(m + 1))


@set_property("fctype", "combiner")
def friedrich_coefficients(x, param):
    """
    Coefficients of polynomial :math:`h(x)`, which has been fitted to 
    the deterministic dynamics of Langevin model

    .. math::
        \dot{x}(t) = h(x(t)) + \mathcal{N}(0,R)

    as described by

        Friedrich et al. (2000): Physics Letters A 271, p. 217-222
        *Extracting model equations from experimental data*


    For short time-series this method is highly dependent on the parameters.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param c: the time series name
    :type c: str
    :param param: contains dictionaries {"coeff": x} with x int and x >= 0
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    coefficients = set([config["coeff"] for config in param])
    for coeff in coefficients:
        if coeff < 0:
            raise ValueError("Coefficients must be positive or zero.")

    m = param[0]['m']
    r = param[0]['r']

    coeff = _estimate_friedrich_coefficients(x, m, r)
    indices = ["m_{}__r_{}__coeff_{}".format(m, r, q) for q in range(m, -1, -1)]

    return zip(indices, coeff)


@set_property("fctype", "simple")
def max_langevin_fixed_point(x, r, m):
    """
    Largest fixed point of dynamics  :math:argmax_x {h(x)=0}` estimated from polynomial :math:`h(x)`, 
    which has been fitted to the deterministic dynamics of Langevin model
    
    .. math::
        \dot(x)(t) = h(x(t)) + R \mathcal(N)(0,1)

    as described by

        Friedrich et al. (2000): Physics Letters A 271, p. 217-222
        *Extracting model equations from experimental data*

    For short time-series this method is highly dependent on the parameters.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param m: order of polynom to fit for estimating fixed points of dynamics
    :type m: int
    :param r: number of quantils to use for averaging
    :type r: float

    :return: Largest fixed point of deterministic dynamics
    :return type: float
    """

    coeff = _estimate_friedrich_coefficients(x, m, r)

    try:
        max_fixed_point = np.max(np.real(np.roots(coeff)))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan
    
    return max_fixed_point


@set_property("fctype", "combiner")
def agg_linear_trend(x, param):
    """
    Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus
    the sequence from 0 up to the number of chunks minus one.

    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.

    The parameters attr controls which of the characteristics are returned. Possible extracted attributes are "pvalue",
    "rvalue", "intercept", "slope", "stderr", see the documentation of linregress for more information.

    The chunksize is regulated by "chunk_len". It specifies how many time series values are in each chunk.

    Further, the aggregation function is controlled by "f_agg", which can use "max", "min" or , "mean", "median"

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"attr": x, "chunk_len": l, "f_agg": f} with x, f an string and l an int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    # todo: we could use the index of the DataFrame here

    calculated_agg = {}
    res_data = []
    res_index = []

    for parameter_combination in param:

        chunk_len = parameter_combination["chunk_len"]
        f_agg = parameter_combination["f_agg"]

        aggregate_result = _aggregate_on_chunks(x, f_agg, chunk_len)
        if f_agg not in calculated_agg or chunk_len not in calculated_agg[f_agg]:
            if chunk_len >= len(x):
                calculated_agg[f_agg] = {chunk_len: np.NaN}
            else:
                lin_reg_result = linregress(range(len(aggregate_result)), aggregate_result)
                calculated_agg[f_agg] = {chunk_len: lin_reg_result}

        attr = parameter_combination["attr"]

        if chunk_len >= len(x):
            res_data.append(np.NaN)
        else:
            res_data.append(getattr(calculated_agg[f_agg][chunk_len], attr))

        res_index.append("f_agg_\"{}\"__chunk_len_{}__attr_\"{}\"".format(f_agg, chunk_len, attr))

    return zip(res_index, res_data)


@set_property("fctype", "combiner")
def energy_ratio_by_chunks(x, param):
    """
    Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the sum of squares over the whole series

    Takes as input parameters the number num_segments of segments to divide the series into and segment_focus
    which is the segment number (starting at zero) to return a feature on.

    Note that the answer for num_segments=1 is a trivial "1" but we handle this scenario
    in case somebody calls it. Sum of the ratios should be 1.0.

    Returns an error for N <= 0

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"num_segments": N, "segment_focus": i} with N, i both ints
    :return: the feature values
    :return feature name, feature value
    """

    res_data = []
    res_index = []
    full_series_energy = np.sum(x ** 2)

    for parameter_combination in param:
        num_segments = parameter_combination["num_segments"]
        segment_focus = parameter_combination["segment_focus"]
        assert segment_focus < num_segments

        segment_length = len(x)//num_segments
        start = segment_focus*segment_length
        end = min((segment_focus+1)*segment_length, len(x))
        res_data.append(np.sum(x[start:end]**2.0)/full_series_energy)
        res_index.append("num_segments_{}__segment_focus_{}".format(num_segments, segment_focus))

    return list(zip(res_index, res_data)) # Materialize as list for Python 3 compatibility with name handling
