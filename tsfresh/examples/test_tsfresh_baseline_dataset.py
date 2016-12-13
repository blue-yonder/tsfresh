# -*- coding: utf-8 -*-
"""
This module implements a function to download a json timeseries data set that is
utilised by tests/baseline/tsfresh_features_test.py to test calculated feature
names and their calculated values are consistent with the known baseline.
"""

from __future__ import absolute_import
from urllib.request import urlopen
import os
from sys import version_info

python_version = int(version_info[0])
module_path = os.path.dirname(__file__)
data_dir = os.path.join(module_path, 'data')
data_file_dir = '%s/test_tsfresh_baseline_dataset' % data_dir
data_file_name = '%s/data.json' % data_file_dir


def download_json_dataset():
    """
    Download the tests baseline timeseries json data set and store it at
    tsfresh/examples/data/test_tsfresh_baseline_dataset/data.json.

    Examples
    ========

    >>> from tsfresh.examples import test_tsfresh_baseline_dataset
    >>> download_json_dataset()
    """

    url = 'https://raw.githubusercontent.com/earthgecko/skyline/master/utils/data.json'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)

    if os.path.isfile(data_file_name):
        return str(data_file_name)

    with urlopen(url) as resp:
        if python_version == 2:
            json_data = resp.read()
        if python_version == 3:
            json_data = resp.read().decode(resp.headers.get_content_charset())
        with open(data_file_name, 'w') as fh:
            fh.write(json_data)

    if os.path.isfile(data_file_name):
        return str(data_file_name)

    return None
