import sys
import os
import shutil
import json
import tempfile
from ast import literal_eval
import re

import unittest
import pandas as pd
from tsfresh import extract_features
from tsfresh import __version__ as tsfresh_version
from tsfresh.examples.test_tsfresh_baseline_dataset import download_json_dataset

# TODO: reconsider including a config with no feature names just declaring the
# current baseline version as there is not difference in the baselines between
# 0.3.0 and 0.3.1, a version should not require a baseline if no changes were
# made, it should just use the last known baseline or just use a specific file
# name for current baseline file and deprecate the old baselines prefixed with
# tsfresh-x.y.z and if __version__ < TSFRESH_BASELINE_VERSION use ah... if each
# one does not have a baseline, which is the last baseline, listdir sort... :)
# Needs thought.

TSFRESH_BASELINE_VERSION = str(tsfresh_version)
if TSFRESH_BASELINE_VERSION == '0.1.1.post0.dev62+ng0f1b4c7':
    # #109 was fixed in 0.3.1, just here for local testing purposes, for the
    # various local version.
    TSFRESH_BASELINE_VERSION = '0.3.0'
if 'post' in TSFRESH_BASELINE_VERSION:
    travis_tsfresh_version = re.sub('\.post.*', '', TSFRESH_BASELINE_VERSION)
    TSFRESH_BASELINE_VERSION = travis_tsfresh_version

python_version = int(sys.version_info[0])
baseline_dir = os.path.dirname(os.path.realpath(__file__))
tests_dir = os.path.dirname(baseline_dir)
tsfresh_dir = os.path.dirname(tests_dir)
baseline_ts_json_file = 'data.json'
baseline_ts_json_baseline = '%s/tsfresh-%s.py%s.%s.features.transposed.csv' % (
    baseline_dir, TSFRESH_BASELINE_VERSION, str(python_version),
    baseline_ts_json_file)
t_fname_out_fail = '%s/tsfresh/examples/data/test_tsfresh_baseline_dataset/tsfresh-unknown-version.py%s.data.json.features.transposed.csv' % (tsfresh_dir, str(python_version))
baseline_ts_json = '%s/tsfresh/examples/data/test_tsfresh_baseline_dataset/data.json' % tsfresh_dir


class TestTsfreshBaseline(unittest.TestCase):
    """
    Test all the features and their calculated values with a 60 data point
    sample of a simple anomalous timeseries data set and compare that the feature
    names and calculated values match the baselines calcualated for the specific
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
    - Modify a feature name or value in your local tests/baseline/tsfresh-<TSFRESH_BASELINE_VERSION>.py<PYTHON_VERSION>.data.json.features.transposed.csv file,
      run the test and either pull it again or revert the change

    """
    def setUp(self):
        self.baseline_ts_json = download_json_dataset()
        self.test_path = tempfile.mkdtemp()
        self.fname_in = '%s/%s' % (self.test_path, baseline_ts_json_file)
        tmp_csv = '%s.tmp.csv' % (self.fname_in)
        t_fname_out = '%s.features.transposed.csv' % self.fname_in

        self.assertTrue(os.path.exists(baseline_ts_json))

        timeseries_json = None
        if os.path.isfile(baseline_ts_json):
            with open(baseline_ts_json, 'r') as f:
                timeseries_json = json.loads(f.read())

        if python_version == 2:
            timeseries_str = str(timeseries_json).replace('{u\'results\': ', '').replace('}', '')
        if python_version == 3:
            timeseries_str = str(timeseries_json).replace('{\'results\': ', '').replace('}', '')

        full_timeseries = literal_eval(timeseries_str)
        timeseries = full_timeseries[:60]
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

        df_features = None
        df = pd.read_csv(tmp_csv, delimiter=',', header=None, names=['metric', 'timestamp', 'value'])
        df.columns = ['metric', 'timestamp', 'value']
        df_features = extract_features(df, column_id='metric', column_sort='timestamp', column_kind=None, column_value=None)

        df_created = None
        # Test the DataFrame
        try:
            df_created = str(df_features.head())
            if df_created:
                self.assertTrue(isinstance(df_created, str))
        # Catch when df_created is None
        except AttributeError:
            self.assertTrue(df_created)
            pass
        # Catch if not defined
        except NameError:
            self.assertTrue(df_created)
            pass

        # Transpose, because we are humans
        df_t = None
        df_t = df_features.transpose()

        # Test the DataFrame
        df_t_created = None
        try:
            df_t_created = str(df_t.head())
            if df_t_created:
                self.assertTrue(isinstance(df_t_created, str))
        # Catch when df_t_created is None
        except AttributeError:
            self.assertTrue(df_t_created)
            pass
        # Catch if not defined
        except NameError:
            self.assertTrue(df_t_created)
            pass

        # Write the transposed csv
        df_t.to_csv(t_fname_out)
        self.df_trans = df_features.transpose()

        self.assertTrue(os.path.isfile(t_fname_out))
        return True

    def tearDown(self):
        # Remove the directory after the test
        ran = False
        fail_msg = 'failed to removed - %s' % self.test_path
        try:
            shutil.rmtree(self.test_path)
            ran = True
        except:
            pass
        self.assertTrue(ran, msg=fail_msg)

    def test_tsfresh_baseline_json(self):
        """
        This test compares the calculated feature names AND values in the
        transposed csv to the feature names AND values in the baseline
        transposed csv.  It outputs the differences if the test fails.
        """
        t_fname_out = '%s.features.transposed.csv' % self.fname_in
        self.assertTrue(os.path.isfile(t_fname_out))

        df_t = pd.read_csv(
            t_fname_out, delimiter=',', header=None,
            names=['feature_name', 'value'])
        df_t_features = []
        for index, line in df_t.iterrows():
            df_t_features.append([str(line[0]), str(line[1])])
        calculated_features = sorted(df_t_features, key=lambda row: row[0], reverse=True)

        df_baseline = pd.read_csv(
            baseline_ts_json_baseline, delimiter=',', header=None,
            names=['feature_name', 'value'])
        df_baseline_features = []
        for index, line in df_baseline.iterrows():
            df_baseline_features.append([str(line[0]), str(line[1])])
        baseline_features = sorted(df_baseline_features, key=lambda row: row[0], reverse=True)

        features_equal = False
        fail_msg = 'none'
        if baseline_features == calculated_features:
            features_equal = True
        else:
            not_in_calculated = [x for x in baseline_features if x not in calculated_features]
            not_in_baseline = [x for x in calculated_features if x not in baseline_features]
            fail_msg = '''
See the docs on how to update the baseline.
New local baseline: %s

NOT in baseline   :: %s

NOT in calculated :: %s''' % (t_fname_out_fail, str(not_in_baseline), str(not_in_calculated))
        if not features_equal:
            shutil.move(t_fname_out, t_fname_out_fail)
        self.assertTrue(features_equal, msg=fail_msg)

if __name__ == '__main__':
    unittest.main()
