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


def _distance(x, y, type="euclid"):
    """
    This calculates the distance metric specified between two 1-D sequences. The default metric is the euclid, the root 
    of the squared differences between components.
    
    :param x: first vector
    :type x: iterable
    :param y: second vector
    :type y: iterable
    :param type: how to calculate the distance
    :type type: str

    :return: the calculated distance between both vectors
    :rtype: float
    """

    x = np.asarray(x)
    y = np.asarray(y)

    assert len(x) == len(y)

    if type.lower() == "euclid":
        return np.linalg.norm(x-y, ord=2)
    else:
        raise ValueError("There is no implementation of a _distance of type {}".format(type))


def _array_of_sliding_windows(data, pattern_length):
    """

    Takes the data, which should be a one dimensional numpy ndarray or pandas Series and constructs a numpy ndarry, 
    which contains subsequences of length pattern_length.

    So lets say you have the array 
    
    [1, 2, 4, 5] 
    
    and want all subsequences of length 3. Calling this method with pattern_length = 3 will return 
    
    [[1, 2, 3], 
    [2, 3, 4], 
    [3, 4, 5]]

    :param data: the vector to look for subsequences in
    :type data: numpy.ndarray or pandas.Series
    :param pattern_length: the length of sliding windows
    :type data: int

    :return: two dimensional array containing the subsequences
    :rtype: numpy.ndarray
    """
    assert isinstance(data, np.ndarray) or isinstance(data, pd.Series)
    assert isinstance(pattern_length, six.integer_types)
    assert len(data.shape) == 1

    dimensions = (len(data) - pattern_length + 1, pattern_length)
    steplen = (data.strides[-1],) + data.strides

    return np.lib.stride_tricks.as_strided(data, shape=dimensions, strides=steplen, writeable=False)


def _match_scores(data, pattern):
    """
    Calculates the distance of the pattern pattern to each subsequence in the sequence data. This method returns a list 
    of distances. 
    
    The distance at position i, is the distance between the pattern and the subsequence that starts at position i in 
    data
    
    :param data: the original time series 
    :type data: iterable
    :param pattern: the pattern to search for
    :type pattern: iterable
    
    :return: the array of distance of pattern to each sliding window
    :rtype: numpy.ndarray
    """
    assert len(data) >= len(pattern)

    res = [_distance(x, pattern) for x in _array_of_sliding_windows(data, len(pattern))]
    return np.array(res)


def _best_n_matches(data, sample, count=1):
    """

    :param data:
    :type data:
    :param sample:
    :type sample:
    :param count:
    :type count:
    
    :return:
    :rtype: list
    """
    match_scores = np.absolute(_array_of_sliding_windows(data, sample))
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
    
    :param length:
    :type length:
    :param candidates:
    :type candidates:
    :param count:
    :type count:
    
    :return:
    :rtype:
    """

    # sort candidates by distance
    candidates.sort(key=lambda result: result[2])

    top_uniques = []
    indexes_of_better_motifs = set()

    for candidate in candidates:

        # compare makes sure candidate doesn't overlay existing top item
        if candidate[0] not in indexes_of_better_motifs:
            top_uniques.append(candidate)
            indexes_of_better_motifs.update(range(candidate[0]-length, candidate[0]+length))

        if len(top_uniques) >= count:
            break
    return top_uniques[0:count]


def _generate_candidates(data, motif_length):
    """
    
    :param data: times series data to match motifs in
    :type data: iterable
    :param motif_length: length of the motifs to look for
    :type motif_length: int
    
    :return: list of candidates
    :rtype: list
    """

    candidates = []
    # todo: Max: wouldn't 2 * motif_length be enough?
    for start in range(len(data) - (3 * motif_length)):
        end = start + motif_length

        pattern = data[start:end]
        pattern_scores = _match_scores(data[end:-motif_length], pattern)

        candidates.append((start,
                           np.argmin(pattern_scores) + end,
                           np.min(pattern_scores)))
    return candidates


def find_motifs(data, motif_length, motif_count, min_data_multiple=8):
    """
    Goes over the data iterable and searches for motifs in it. The result is a list of tuples holding the start_point
    of each motif, the best match point, and its distance

    :param data: times series data to match motifs in
    :type data: iterable
    :param motif_length: length of the motifs to look for
    :type motif_length: int
    :param motif_count: how many motifs to return
    :type motif_count: int
    :param min_data_multiple: 
    :rtype min_data_multiple

    :return: tuples of length 3 holding start_point of each motif, best match point, and distance
    :return type: list
    """

    if motif_length * min_data_multiple > len(data):
        raise ValueError("Motif size too large for dataset.")

    candidates = _generate_candidates(data, motif_length)

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

    # Todo: Max, maybe we should turn off the backmatching, see the following comment
    # It's interesting that this can return values for distance measures better than the "best" found during motif
    # finding. I think this is back matching,

    l = len(motif)
    pattern = data[motif[0]:motif[0] + l]

    pattern_scores = _match_scores(data, pattern)
    pattern_scores[motif[0] - l:motif[0] + l] = np.inf
    return int(sum(pattern_scores < dist))
