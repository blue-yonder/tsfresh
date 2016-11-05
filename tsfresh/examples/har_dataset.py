# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

"""
This module implements functions to download and load Human Activity Recognition dataset. 

https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
"""

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd


def download_har_dataset():
    """
    Download the human activity recognition dataset from the UCI Machine Learning Repository and store it locally.
    
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

    return pd.read_csv('data/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt', delim_whitespace=True, header=None)