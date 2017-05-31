# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

"""
This module implements functions to download the Robot Execution Failures LP1 Data Set[1] and load it as as DataFrame.

*Important:* You need to download the data set yourself, either manually or via the function
:func:`~tsfresh.examples.robot_execution_failures.download_robot_execution_failures`

References
----------
.. [1] http://mlr.cs.umass.edu/ml/datasets/Robot+Execution+Failures
.. [2] Lichman, M. (2013).
    UCI Machine Learning Repository [http://mlr.cs.umass.edu/ml].
    Irvine, CA: University of California, School of Information and Computer Science.
.. [3] Camarinha-Matos, L.M., L. Seabra Lopes, and J. Barata (1996).
    Integration and Learning in Supervision of Flexible Assembly Systems.
    "IEEE Transactions on Robotics and Automation", 12 (2), 202-219

"""

from __future__ import absolute_import, division

from builtins import map
import os
import pandas as pd
import requests
import logging

_logger = logging.getLogger(__name__)

UCI_MLD_REF_MSG = ("The example data could not be found. You need to download the Robot Execution Failures "
                   "LP1 Data Set from the UCI Machine Learning Repository. To do so, you can call the function "
                   "tsfresh.examples.robot_execution_failures.download_robot_execution_failures")
UCI_MLD_REF_URL = "https://raw.githubusercontent.com/MaxBenChrist/robot-failure-dataset/master/lp1.data.txt"

module_path = os.path.dirname(__file__)
data_file_name = os.path.join(module_path, 'data', 'robotfailure-mld', 'lp1.data')


def download_robot_execution_failures():
    """
    Download the Robot Execution Failures LP1 Data Set[1] from the UCI Machine Learning Repository[2] and store it locally.
    :return:

    Examples
    ========

    >>> from tsfresh.examples import download_robot_execution_failures
    >>> download_robot_execution_failures()
    """
    if os.path.exists(data_file_name):
        _logger.warning("You have already downloaded the Robot Execution Failures LP1 Data Set.")
        return

    if not os.access(module_path, os.W_OK):
        raise RuntimeError("You don't have the necessary permissions to download the Robot Execution Failures LP1 Data "
                           "Set into the module path. Consider installing the module in a virtualenv you "
                           "own or run this function with appropriate permissions.")

    os.makedirs(os.path.dirname(data_file_name))

    r = requests.get(UCI_MLD_REF_URL)

    if r.status_code != 200:
        raise RuntimeError("Could not download the Robot Execution Failures LP1 Data Set from the UCI Machine Learning "
                           "Repository. HTTP status code: {}".format(r.status_code))

    with open(data_file_name, "w") as f:
        f.write(r.text)


def load_robot_execution_failures():
    """
    Load the Robot Execution Failures LP1 Data Set[1].
    The Time series are passed as a flat DataFrame.

    Examples
    ========

    >>> from tsfresh.examples import load_robot_execution_failures
    >>> df, y = load_robot_execution_failures()
    >>> print(df.shape)
    (1320, 8)

    :return: time series data as :class:`pandas.DataFrame` and target vector as :class:`pandas.Series`
    :rtype: tuple
    """
    if not os.path.exists(data_file_name):
        raise RuntimeError(UCI_MLD_REF_MSG)

    id_to_target = {}
    df_rows = []

    with open(data_file_name) as f:
        cur_id = 0
        time = 0

        for line in f.readlines():
            # New sample --> increase id, reset time and determine target
            if line[0] not in ['\t', '\n']:
                cur_id += 1
                time = 0
                if line.strip() == 'normal':
                    id_to_target[cur_id] = 0
                else:
                    id_to_target[cur_id] = 1
            # Data row --> split and convert values, create complete df row
            elif line[0] == '\t':
                values = list(map(int, line.split('\t')[1:]))
                df_rows.append([cur_id, time] + values)
                time += 1

    df = pd.DataFrame(df_rows, columns=['id', 'time', 'a', 'b', 'c', 'd', 'e', 'f'])
    y = pd.Series(id_to_target)

    return df, y
