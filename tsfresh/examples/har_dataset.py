# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

"""
This module implements functions to download and load the Human Activity Recognition dataset [4]_.
A description of the data set can be found in [5]_.


References
----------

.. [4] https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
.. [5] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. (2013)
        A Public Domain Dataset for Human Activity Recognition Using Smartphones.
        21th European Symposium on Artificial Neural Networks,
        Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

"""

import logging
import os
import shutil
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests

_logger = logging.getLogger(__name__)

module_path = os.path.dirname(__file__)
data_file_name = os.path.join(module_path, "data", "UCI HAR Dataset")


def download_har_dataset(folder_name=data_file_name):
    """
    Download human activity recognition dataset from UCI ML Repository and store it at /tsfresh/notebooks/data.

    Examples
    ========

    >>> from tsfresh.examples import har_dataset
    >>> har_dataset.download_har_dataset()
    """

    zipurl = "https://github.com/MaxBenChrist/human-activity-dataset/blob/master/UCI%20HAR%20Dataset.zip?raw=True"

    if not os.access(module_path, os.W_OK):
        raise RuntimeError(
            "You don't have the necessary permissions to download the Human Activity Dataset "
            "Set into the module path. Consider installing the module in a virtualenv you "
            "own or run this function with appropriate permissions."
        )

    if os.path.exists(os.path.join(folder_name, "UCI HAR Dataset")):
        _logger.warning("You have already downloaded the Human Activity Data Set.")
        return

    os.makedirs(folder_name, exist_ok=True)

    r = requests.get(zipurl, stream=True)
    if r.status_code != 200:
        raise RuntimeError(
            "Could not download the Human Activity Data Set from GitHub."
            "HTTP status code: {}".format(r.status_code)
        )

    with ZipFile(BytesIO(r.content)) as zfile:
        zfile.extractall(path=folder_name)


def load_har_dataset(folder_name=data_file_name):
    data_file_name_dataset = os.path.join(
        folder_name,
        "UCI HAR Dataset",
        "train",
        "Inertial Signals",
        "body_acc_x_train.txt",
    )
    try:
        return pd.read_csv(data_file_name_dataset, delim_whitespace=True, header=None)
    except OSError:
        raise OSError(
            "File {} was not found. Have you downloaded the dataset with download_har_dataset() "
            "before?".format(data_file_name_dataset)
        )


def load_har_classes(folder_name=data_file_name):
    data_file_name_classes = os.path.join(
        folder_name, "UCI HAR Dataset", "train", "y_train.txt"
    )
    try:
        return pd.read_csv(
            data_file_name_classes, delim_whitespace=True, header=None, squeeze=True
        )
    except OSError:
        raise OSError(
            "File {} was not found. Have you downloaded the dataset with download_har_dataset() "
            "before?".format(data_file_name_classes)
        )
