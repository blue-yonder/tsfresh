# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

"""
Functions and support for finding, grooming, and filtering time series motifs.
A motif is a characteristic subsequence of the inspected time series.

by Ezekiel Kruglick
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import pandas as pd
from six.moves import range
import six


def distance(x, y, type="euclid"):
    """
    This calculates the distance metric between two 1-D sequences. Default below is the num of squares, override this
    function to use another metric

    :param x: first vector
    :type x: iterable
    :param y: second vector
    :type y: iterable
    :param type: how to calculate the distance
    :type type: str

    :return: the calculated distance between both vectors
    """

    x = np.asarray(x)
    y = np.asarray(y)

    assert len(x) == len(y)

    if type.lower() == "euclid":
        return np.linalg.norm(x-y, ord=2)
    else:
        raise ValueError("Have not implemented distance of type {}".format(type))


def _sliding_window(data, pattern_length):
    """

    Takes the data, which should be a one dimensional numpy ndarray or pandas Series and constructs a numpy view on
    containing the subsequences of length pattern_length.

    So lets say you have the array [1, 2, 4, 5] and want all subsequences of length 3. Calling this method with
    pattern_length = 3 will return [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

    :param data: the vector to look for subsequences in
    :type data: numpy.ndarray or pandas.Series

    :param pattern_length: the length of sliding windows
    :type data: int

    :return: two dimensional array
    """
    assert isinstance(data, np.ndarray) or isinstance(data, pd.Series)
    assert isinstance(pattern_length, six.integer_types)
    assert len(data.shape) == 1

    # we will create a view on the original array, we do not override the original array, so we make a copy
    data = np.copy(data)

    dimensions = (len(data) - pattern_length + 1, pattern_length)
    steplen = (data.strides[-1],) + data.strides

    # TODO: can we somehow remove the dependence on `as_strided`? even its own docstrings warns not to use it...
    return np.lib.stride_tricks.as_strided(data, shape=dimensions, strides=steplen)


def _match_scores(data, pattern):
    """

    :param data:
    :param pattern:
    :return:
    """

    return np.array([distance(x, pattern) for x in _sliding_window(data, len(pattern))])


def _best_n_matches(data, sample, count=1):
    """

    :param data:
    :param sample:
    :param count:
    :return:
    """
    match_scores = np.absolute(_sliding_window(data, sample))
    top_list = []
    for _ in range(count):
        top_spot = np.argmax(match_scores)
        match_scores[top_spot:top_spot + len(sample)] = np.zeros((len(sample, )))
        top_list.append(top_spot)
    return top_list


def _candidates_top_uniques(length, candidates, count):
    """
    The candidate filter if statement first makes sure this isn't a reverse match (A-->B, don't also return B-->A),
    then takes the top item of any overlapping motifs (first ones in are the best ones, then exclude lower ones),
    then eliminates any matches that contain parts of the better ones above

    :param candidates:
    :param count:
    :return:
    """

    # sort candidates by distance
    candidates.sort(key=lambda result: result[2])

    top_uniques = []
    for candidate in candidates:

        # todo: the lists should not be constructed from scratch in every iteration, maybe work with append?
        if          candidate[0] not in [y for x in top_uniques for y in range(x[1],          x[1] + length)] \
                and candidate[0] not in [y for x in top_uniques for y in range(x[0] - length, x[0] + length)]:

            top_uniques.append(candidate)
        if len(top_uniques) >= count:
            break
    return top_uniques[0:count]


def find_motifs(data, motif_length, motif_count):
    """
    Goes over the data iterable and searches for motifs in it. The result is a list of tuples holding the start_point
    of each motif, the best match point, and its distance

    :param data: times series data to match motifs in
    :type data: iterable
    :param motif_length: length of the motifs to look for
    :type motif_length: int
    :param motif_count: how many motifs to return
    :type motif_count: int

    :return: tuples of length 3 holding start_point of each motif, best match point, and distance
    :return type: list
    """

    # todo: why the factor 8? why not 7 or 6
    if motif_length * 8 > len(data):
        raise ValueError("Motif size too large for dataset.")

    candidates = []

    # todo: wouldn't 2 * motif_length be enough?
    for start in range(len(data) - (3 * motif_length)):

        end = start + motif_length

        pattern = data[start:end]
        # todo: why you are not matching backwards? You only look for matches of the pattern starting at start
        pattern_scores = _match_scores(data[end:-motif_length], pattern)

        candidates.append((start,
                           np.argmin(pattern_scores) + end,
                           np.min(pattern_scores)))

    return _candidates_top_uniques(motif_length, candidates, motif_count)


def count_motifs(data, motif, dist=10):
    """

    :param data: time series data to search
    :type data: iterable
    :param motif: motif in tuple format (start, best match, score)
    :type motif: list of tuples
    :param dist: Count any segment found to be this close
    :type dist: numeric

    :return: returns an integer count
    :return type: int
    """

    # Todo: turn off the backmatching, see following comment
    # It's interesting that this can return values for distance measures better than the "best" found during motif
    # finding. I think this is back matching,

    l = len(motif)
    pattern = data[motif[0]:motif[0] + l]

    pattern_scores = _match_scores(data, pattern)
    pattern_scores[motif[0] - l:motif[0] + l] = np.inf
    return np.sum(pattern_scores < dist)
