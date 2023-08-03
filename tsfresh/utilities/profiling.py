# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
Contains methods to start and stop the profiler that checks the runtime of the different feature calculators
"""

import cProfile
import io
import logging
import pstats

from tsfresh import defaults

_logger = logging.getLogger(__name__)


# todo: tackle a debate about the need for this profiler
# todo: we need unit tests for the profiling routine


def start_profiling():
    """
    Helper function to start the profiling process and return the profiler (to close it later).

    :return: a started profiler.
    :rtype: cProfile.Profile

    Start and stop the profiler with:

    >>> profiler = start_profiling()
    >>> # Do something you want to profile
    >>> end_profiling(profiler, "cumulative", "out.txt")
    """
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler


def end_profiling(profiler, filename, sorting=None):
    """
    Helper function to stop the profiling process and write out the profiled
    data into the given filename. Before this, sort the stats by the passed sorting.

    :param profiler: An already started profiler (probably by start_profiling).
    :type profiler: cProfile.Profile
    :param filename: The name of the output file to save the profile.
    :type filename: basestring
    :param sorting: The sorting of the statistics passed to the sort_stats function.
    :type sorting: basestring

    :return: None
    :rtype: None

    Start and stop the profiler with:

    >>> profiler = start_profiling()
    >>> # Do something you want to profile
    >>> end_profiling(profiler, "out.txt", "cumulative")
    """
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(sorting)
    ps.print_stats()

    with open(filename, "w+") as f:
        _logger.info(
            "[calculate_ts_features] Finished profiling of time series feature extraction"
        )
        f.write(s.getvalue())


def get_n_jobs():
    """
    Get the number of jobs to use for parallel processing.

    :return: The number of jobs to use for parallel processing.
    :rtype: int
    """
    return defaults.N_PROCESSES


def set_n_jobs(n_jobs):
    """
    Set the number of jobs to use for parallel processing.

    :param n_jobs: The number of jobs to use for parallel processing.
    :type n_jobs: int

    :return: None
    :rtype: None
    """
    defaults.N_PROCESSES = n_jobs
