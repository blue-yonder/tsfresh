# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

"""
This module implements functions to download and load the Human Activity Recognition dataset [4].
A description of the data set can be found in [5].


References
----------

.. [4] https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
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


def download_har_dataset():
    """
    Download human activity recognition dataset from UCI ML Repository and store it at /tsfresh/notebooks/data.
    
    Examples
    ========

    >>> from tsfresh.examples import har_dataset
    >>> download_har_dataset()
    """
    
    zipurl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    
    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall('data')
        zfile.close()


def load_har_dataset():
    return pd.read_csv('data/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt', 
        delim_whitespace=True, header=None)


def load_har_classes():
    return pd.read_csv('data/UCI HAR Dataset/train/y_train.txt', 
        delim_whitespace=True, header=None)