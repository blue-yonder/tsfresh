import numpy as np
"""
Functions and support for finding, grooming, and filtering time series data motifs
by Ezekiel Kruglick
"""


def distance(x, y):
    """
    This calculates the distance metric between two 1-D sequences. Default below is the num of squares, override this
    function to use another metric
    :param x: first single dimensional data (list, 1-D numpy array)
    :param y: second single dimensional data
    :return: distance metric between the two data parameters
    """
    return np.sqrt(np.sum((x - y) ** 2))


def _sliding_window(data, sample_length):
    assert (isinstance(sample_length, int))
    dimensions = (data.shape[-1] - sample_length + 1, sample_length) + data.shape[:-1]
    steplen = (data.strides[-1],) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=dimensions, strides=steplen)


def _match_scores(data, match_pattern):
    return np.array([distance(x, match_pattern) for x in _sliding_window(data, len(match_pattern))])


def _best_n_matches(data, sample, count=1):
    match_scores = np.absolute(_sliding_window(data, sample))
    top_list = []
    for i in range(count):
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
    candidates.sort(key=lambda result: result[2])
    top_uniques = []
    for candidate in candidates:
        if candidate[0] not in [y for x in top_uniques for y in range(x[1], x[1] + length)] \
                and candidate[0] not in [y for x in top_uniques for y in
                                         range(x[0] - length, x[0] + length)]:
            top_uniques.append(candidate)
        if len(top_uniques) >= count:
            break
    return top_uniques[0:count]


def find_motifs(length, data, motif_count):
    """

    :param length: length of the motifs to look for
    :param data: times series data to match motifs in
    :param motif_count: how many motifs to return
    :return: tuples of length 3 holding start_point of each motif, best match point, and distance metric
    """
    if length * 8 > len(data):
        raise ValueError("Motif size too large for dataset.")
    candidates = []
    for start in xrange(len(data) - (3 * length)):
        match_pattern = data[start:start + length]
        pattern_scores = _match_scores(data[start + length:-length], match_pattern)
        candidates.append((start,
                           np.argmin(pattern_scores) + start + (2 * length),
                           np.min(pattern_scores) + start + (2 * length)))
    return _candidates_top_uniques(length, candidates, motif_count)


def count_motifs(data, motif, dist=10):
    """
    It's interesting that this can return values for distance measures better than the "best" found during motif
    finding. I think this is backmatching,
    :param data: time series data to search
    :param motif: motif in tuple format (start, best match, score)
    :param dist: Count any segment found to be this close
    :return: returns an integer count
    """
    length = len(motif)
    match_pattern = data[motif[0]:motif[0] + length]
    pattern_scores = _match_scores(data, match_pattern)
    pattern_scores[motif[0] - length:motif[0] + length] = np.inf
    return np.sum(pattern_scores < dist)
