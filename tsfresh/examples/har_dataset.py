# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

"""
<<<<<<< HEAD
This module implements functions to download and load Human Activity Recognition dataset. 

https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
"""

=======
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
>>>>>>> upstream/master
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd
<<<<<<< HEAD


def download_har_dataset():
    """
    Download human activity recognition dataset from UCI ML Repository and store it in /tsfresh/notebooks/data.
=======
import os

module_path = os.path.dirname(__file__)
data_file_name = os.path.join(module_path, 'data')
data_file_name_dataset = os.path.join(module_path, 'data', 'UCI HAR Dataset', 'train', 'Inertial Signals',
                                      'body_acc_x_train.txt')
data_file_name_classes = os.path.join(module_path, 'data','UCI HAR Dataset', 'train', 'y_train.txt')

def download_har_dataset():
    """
    Download human activity recognition dataset from UCI ML Repository and store it at /tsfresh/notebooks/data.
>>>>>>> upstream/master
    
    Examples
    ========

    >>> from tsfresh.examples import har_dataset
    >>> download_har_dataset()
    """
    
    zipurl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    
    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
<<<<<<< HEAD
            zfile.extractall('data')
=======
            zfile.extractall(path=data_file_name)
>>>>>>> upstream/master
        zfile.close()


def load_har_dataset():
<<<<<<< HEAD
    return pd.read_csv('data/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt', 
        delim_whitespace=True, header=None)


def load_har_classes():
    return pd.read_csv('data/UCI HAR Dataset/train/y_train.txt', 
        delim_whitespace=True, header=None)
=======
    try:
        return pd.read_csv(data_file_name_dataset, delim_whitespace=True, header=None)
    except IOError:
        raise IOError('File {} was not found. Have you downloaded the dataset with download_har_dataset() '
                       'before?'.format(data_file_name_dataset))


def load_har_classes():
    try:
        return pd.read_csv(data_file_name_classes, delim_whitespace=True, header=None)
    except IOError:
        raise IOError('File {} was not found. Have you downloaded the dataset with download_har_dataset() '
                       'before?'.format(data_file_name_classes))
>>>>>>> upstream/master
