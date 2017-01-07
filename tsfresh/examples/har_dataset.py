# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

"""
This module implements functions to download and load the Human Activity Recognition dataset [4].
A description of the data set can be found in [5].


References
----------

.. [4] http://mlr.cs.umass.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
.. [5] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. (2013)
     A Public Domain Dataset for Human Activity Recognition Using Smartphones.
     21th European Symposium on Artificial Neural Networks,
     Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

"""

from __future__ import absolute_import, division
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd
import os
import logging

_logger = logging.getLogger(__name__)

module_path = os.path.dirname(__file__)
data_file_name = os.path.join(module_path, 'data')
data_file_name_dataset = os.path.join(module_path, 'data', 'UCI HAR Dataset', 'train', 'Inertial Signals',
                                      'body_acc_x_train.txt')
data_file_name_classes = os.path.join(module_path, 'data', 'UCI HAR Dataset', 'train', 'y_train.txt')


def download_har_dataset():
    """
    Download human activity recognition dataset from UCI ML Repository and store it at /tsfresh/notebooks/data.
    
    Examples
    ========

    >>> from tsfresh.examples import har_dataset
    >>> download_har_dataset()
    """

    zipurl = 'https://github.com/MaxBenChrist/human-activity-dataset/blob/master/UCI%20HAR%20Dataset.zip?raw=true'

    if os.path.exists(data_file_name_dataset) and os.path.exists(data_file_name_classes):
        _logger.warning("You have already downloaded the Human Activity Data Set.")
        return

    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(path=data_file_name)
        zfile.close()


def load_har_dataset():
    try:
        return pd.read_csv(data_file_name_dataset, delim_whitespace=True, header=None)
    except IOError:
        raise IOError('File {} was not found. Have you downloaded the dataset with download_har_dataset() '
                      'before?'.format(data_file_name_dataset))


def load_har_classes():
    try:
        return pd.read_csv(data_file_name_classes, delim_whitespace=True, header=None, squeeze=True)
    except IOError:
        raise IOError('File {} was not found. Have you downloaded the dataset with download_har_dataset() '
                      'before?'.format(data_file_name_classes))
