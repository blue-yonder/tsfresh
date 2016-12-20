# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This module contains code to comapre results of tsfresh to calculated older snapshots
"""

import sys
import os
import shutil
import json
import tempfile
import re
import unittest
import pandas as pd
from tsfresh import extract_features
from tsfresh import __version__ as tsfresh_version
from tsfresh.examples.test_tsfresh_baseline_dataset import download_json_dataset
from numpy.testing import assert_almost_equal

# TODO: reconsider including a config with no feature names just declaring the
# current baseline version as there is not difference in the baselines between
# 0.3.0 and 0.3.1, a version should not require a baseline if no changes were
# made, it should just use the last known baseline or just use a specific file
# name for current baseline file and deprecate the old baselines prefixed with
# tsfresh-x.y.z and if __version__ < TSFRESH_BASELINE_VERSION use ah... if each
# one does not have a baseline, which is the last baseline, listdir sort... :)
# Needs thought.

# version number stuff, getting python and tsfresh version
TSFRESH_BASELINE_VERSION = str(tsfresh_version)
PYTHON_VERSION = str(int(sys.version_info[0]))

# todo: what is the point of this checks ?
if TSFRESH_BASELINE_VERSION == '0.1.1.post0.dev62+ng0f1b4c7':
    # #109 was fixed in 0.3.1, just here for local testing purposes, for the
    # various local version.
    TSFRESH_BASELINE_VERSION = '0.3.0'

# todo: what is the point of this check ?
if 'post' in TSFRESH_BASELINE_VERSION:
    TSFRESH_BASELINE_VERSION = re.sub('\.post.*', '', TSFRESH_BASELINE_VERSION)

if 'dirty' in TSFRESH_BASELINE_VERSION:
    # remove "dirty" part of version tag, for example 0.4.0+ng7e4c397.dirty becomes 0.4.0
    TSFRESH_BASELINE_VERSION = TSFRESH_BASELINE_VERSION.rsplit('+', 1)[0]

# IO stuff, getting directories to read and to write
baseline_dir = os.path.dirname(os.path.realpath(__file__))
tests_dir = os.path.dirname(baseline_dir)
tsfresh_dir = os.path.dirname(tests_dir)

baseline_ts_json_baseline = '{}/tsfresh-{}.py{}.data.json.features.transposed.csv'.format(baseline_dir,
                                                                                          TSFRESH_BASELINE_VERSION,
                                                                                          PYTHON_VERSION)
t_fname_out_fail = '{}/tsfresh/examples/data/test_tsfresh_baseline_dataset/' \
                   'tsfresh-unknown-version.py{}s.data.json.features.transposed.csv'.format(tsfresh_dir, PYTHON_VERSION)
baseline_ts_json = '%s/tsfresh/examples/data/test_tsfresh_baseline_dataset/data.json' % tsfresh_dir


class TestTsfreshBaseline(unittest.TestCase):
    """
    Test all the features and their calculated values with a 60 data point
    sample of a simple anomalous timeseries data set and compare that the feature
    names and calculated values match the baselines calculated for the specific
    version of tsfresh.

    .. warning:: the Python 2 and 3 calculate different results in terms of
        float precision therefore baseline transposed features csv files are
        required for both py2 and py3.

    Running the test
    ================

    .. code-block:: bash

        cd "<YOUR_TSFRESH_DIR>"
        python -m pytest tests/baseline/tsfresh_features_test.py


    Test the test fails
    ===================

    To test that the test fails as desired and as does what it is supposed to do
    there are 2 methods to achieve this:

    - Modify the first value in your local tsfresh/examples/data/test_tsfresh_baseline_dataset/data.json
      and run the test, then delete the modified local data.json file to be pulled down again.
    - Modify a feature name or value in your local
         tests/baseline/tsfresh-<TSFRESH_BASELINE_VERSION>.py<PYTHON_VERSION>.data.json.features.transposed.csv file,
      run the test and either pull it again or revert the change

    """

    def setUp(self):
        download_json_dataset()

        self.test_path = tempfile.mkdtemp()
        self.fname_in = '%s/%s' % (self.test_path, 'data.json')
        tmp_csv = '%s.tmp.csv' % (self.fname_in)
        t_fname_out = '%s.features.transposed.csv' % self.fname_in

        self.assertTrue(os.path.isfile(baseline_ts_json))

        with open(baseline_ts_json, 'r') as f:
            timeseries_json = json.loads(f.read())

        timeseries = timeseries_json["results"][:60]
        self.assertEqual(int(timeseries[0][0]), 1369677886)
        self.assertEqual(len(timeseries), 60)

        for ts, value in timeseries:
            metric = 'tsfresh_features_test'
            timestamp = int(ts)
            value = str(float(value))
            utc_ts_line = '%s,%s,%s\n' % (metric, str(timestamp), value)
            with open(tmp_csv, 'a') as fh:
                fh.write(utc_ts_line)

        self.assertTrue(os.path.isfile(tmp_csv))

        df = pd.read_csv(tmp_csv, delimiter=',', header=None, names=['metric', 'timestamp', 'value'])
        df.columns = ['metric', 'timestamp', 'value']
        df_features = extract_features(df,
                                       column_id='metric', column_sort='timestamp', column_kind=None, column_value=None)

        self.assertTrue(str(df_features.head()))

        # Transpose, because we are humans
        self.df_trans = df_features.transpose()

        # Test the DataFrame
        self.assertTrue(str(self.df_trans.head()))

        # Write the transposed csv
        self.df_trans.to_csv(t_fname_out)

        self.assertTrue(os.path.isfile(t_fname_out))

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_path)

    def test_tsfresh_baseline_json(self):
        """
        This test compares the calculated feature names AND values in the
        transposed csv to the feature names AND values in the baseline
        transposed csv.  It outputs the differences if the test fails.
        """
        t_fname_out = '%s.features.transposed.csv' % self.fname_in
        self.assertTrue(os.path.isfile(t_fname_out))

        df_t = pd.read_csv(t_fname_out, delimiter=',', header=None, names=['feature_name', 'value']
                           ).sort_values(by="feature_name").reset_index(drop=True)

        calculated_features = set(df_t["feature_name"])

        df_baseline = pd.read_csv(baseline_ts_json_baseline, delimiter=',', header=None, names=['feature_name', 'value']
                                  ).sort_values(by="feature_name").reset_index(drop=True)

        baseline_features = set(df_baseline["feature_name"])

        try:
            not_in_calculated = baseline_features - calculated_features
            not_in_baseline = calculated_features - baseline_features

            self.assertEqual(not_in_calculated, set())
            self.assertEqual(not_in_baseline, set())

            for t_row, baseline_row in zip(df_t.itertuples(), df_baseline.itertuples()):
                if t_row.value != "tsfresh_features_test":
                    assert_almost_equal(float(t_row.value), float(baseline_row.value))
                    self.assertEqual(t_row.feature_name, baseline_row.feature_name)

        except AssertionError:
            shutil.move(t_fname_out, t_fname_out_fail)
            print("See the docs on how to update the baseline."
                  "New local baseline: {local_baseline}".format(local_baseline=t_fname_out_fail))
            raise


if __name__ == '__main__':
    unittest.main()
