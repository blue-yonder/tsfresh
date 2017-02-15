# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
Some helper functions.
"""

import logging

_logger = logging.getLogger(__name__)


def calculate_best_chunksize(iterable_list, settings):
    """
    Helper function to calculate the best chunksize for a given number of elements to calculate,
    or use the one in the settings object.

    The formula is more or less an empirical result.
    :param iterable_list: A list which defines how many calculations there need to be.
    :param settings: The settings object where the chunksize may already be given (or not).
    :return: The chunksize which should be used.

    TODO: Investigate which is the best chunk size for different settings.
    """
    if not settings.chunksize:
        chunksize, extra = divmod(len(iterable_list), settings.n_processes * 5)
        if extra:
            chunksize += 1
    else:
        chunksize = settings.chunksize
    return chunksize