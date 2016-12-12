import sys
import os
from time import time, sleep
import shutil
import datetime
import csv
import json
import errno
import tempfile

import unittest
from mock import Mock, patch
import numpy as np
import pandas as pd
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.examples.test_tsfresh_baseline_dataset import download_json_dataset

from tsfresh_feature_names import TSFRESH_FEATURES, TSFRESH_BASELINE_VERSION

python_version = int(sys.version_info[0])
baseline_dir = os.path.dirname(os.path.realpath(__file__))
tests_dir = os.path.dirname(baseline_dir)
tsfresh_dir = os.path.dirname(tests_dir)
baseline_ts_json_file = 'data.json'
baseline_ts_json_baseline = '%s/tsfresh-%s.py%s.%s.features.transposed.csv' % (
    baseline_dir, TSFRESH_BASELINE_VERSION, str(python_version),
    baseline_ts_json_file)
t_fname_out_fail = '%s/tsfresh/examples/data/test_tsfresh_baseline_dataset/tsfresh-unknown-version.py%s.data.json.features.transposed.csv' % (tsfresh_dir, str(python_version))
test_path = tempfile.mkdtemp()
fname_in = '%s/%s' % (test_path, baseline_ts_json_file)
tmp_csv = '%s.tmp.csv' % (fname_in)
fname_out = '%s.features.csv' % fname_in
t_fname_out = '%s.features.transposed.csv' % fname_in


class TestTsfreshBaseline(unittest.TestCase):
    """
    Test all the features and their calculated values with a 60 data point
    sample of simple anomalous timeseries data set and compare that the feature
    names and calculated values match the baselines calcualated for the specific
    verison of tsfresh as defined by :mod:`tsfresh_feature_names.TSFRESH_FEATURES`

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

    def tearDown(self):
        # Remove the directory after the test
        try:
            shutil.rmtree(test_path)
        except:
            pass

    def test_tsfresh_baseline(self):
        """
        This test should generates a set of resources from a sample 60 data
        point timeseries, extracts the features and saves a transposed csv of
        feature_name,value that is used to compare against to baseline
        transposed csv
        """
        baseline_ts_json = self.baseline_ts_json

        test_path_created = False
        if os.path.exists(test_path):
            test_path_created = True
        self.assertEqual(test_path_created, True)

        json_data_exists = False
        if os.path.isfile(baseline_ts_json):
            json_data_exists = True

        self.assertEqual(json_data_exists, True)

        timeseries_json = None
        if os.path.isfile(baseline_ts_json):
            with open(baseline_ts_json, 'r') as f:
                timeseries_json = json.loads(f.read())

        if python_version == 2:
            timeseries_str = str(timeseries_json).replace('{u\'results\': ', '').replace('}', '')
        if python_version == 3:
            timeseries_str = str(timeseries_json).replace('{\'results\': ', '').replace('}', '')

        full_timeseries = eval(timeseries_str)
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

        if os.path.isfile(tmp_csv):
            file_created = True
        else:
            file_created = False
        self.assertEqual(file_created, True)

        df_features = None
        if os.path.isfile(t_fname_out):
            return True
        df = pd.read_csv(tmp_csv, delimiter=',', header=None, names=['metric', 'timestamp', 'value'])
        df.columns = ['metric', 'timestamp', 'value']
        df_features = extract_features(df, column_id='metric', column_sort='timestamp', column_kind=None, column_value=None)

        df_created = None
        df_assert = None
        # Test the DataFrame
        try:
            df_created = str(df_features.head())
            if df_created:
                df_assert = True
                self.assertEqual(df_assert, True)
        # Catch when df_created is None
        except AttributeError:
            self.assertEqual(df_created, True)
            pass
        # Catch if not defined
        except NameError:
            self.assertEqual(df_created, True)
            pass

        if os.path.isfile(t_fname_out):
            return True

        # Write to disk
        df_features.to_csv(fname_out)

        if os.path.isfile(fname_out):
            file_created = True
        else:
            file_created = False
        self.assertEqual(file_created, True)

        if os.path.isfile(t_fname_out):
            return True

        # Transpose
        df_t = None
        df_t = df_features.transpose()
        # df_t.sort_values(by=[0], ascending=[False])

        # Test the DataFrame
        df_t_created = None
        df_t_assert = None
        try:
            df_t_created = str(df_t.head())
            if df_t_created:
                df_t_assert = True
                self.assertEqual(df_t_assert, True)
        # Catch when df_t_created is None
        except AttributeError:
            self.assertEqual(df_t_created, True)
            pass
        # Catch if not defined
        except NameError:
            self.assertEqual(df_t_created, True)
            pass

        # Write the transposed csv
        df_t.to_csv(t_fname_out)

        if os.path.isfile(t_fname_out):
            file_created = True
        else:
            file_created = False
        self.assertEqual(file_created, True)
        return True

    def test_tsfresh_features(self):
        """
        This test compares the calculated feature names in the transposed csv to
        the feature names in the baseline transposed csv and will display new
        or missing feature names if it is fails.
        """
        start = self.test_tsfresh_baseline()

        feature_names_determined = False
        feature_names = []
        if start:
            count_id = 0
            with open(t_fname_out, 'rt') as fr:
                reader = csv.reader(fr, delimiter=',')
                for i, line in enumerate(reader):
                    if str(line[0]) != '':
                        if ',' in line[0]:
                            feature_name = '"%s"' % str(line[0])
                        else:
                            feature_name = str(line[0])
                        count_id += 1
                        feature_names.append([count_id, feature_name])

        if feature_names != []:
            feature_names_determined = True

        fail_msg = 'tsfresh feature names were not determined from %s' % t_fname_out
        self.assertEqual(feature_names_determined, True, msg=fail_msg)

        max_known_id = int(TSFRESH_FEATURES[-1][0])
        valid_max_id = None
        fail_msg = 'max_known_id not an integer'
        try:
            _valid_max_id = max_known_id + 1
            valid_max_id = True
        except:
            valid_max_id = False
        self.assertEqual(valid_max_id, True, msg=fail_msg)

        max_seen_id = int(feature_names[-1][0])
        valid_max_seen_id = None
        fail_msg = 'max_seen_id not an integer'
        try:
            _valid_max_seen_id = max_seen_id + 1
            valid_max_seen_id = True
        except:
            valid_max_seen_id = False
        self.assertEqual(valid_max_seen_id, True, msg=fail_msg)

        feature_names_match = False
        if feature_names != TSFRESH_FEATURES:

            def getKey(item):
                return item[0]

            sorted_feature_names = sorted(feature_names, key=getKey)
            sorted_tsfresh_features = sorted(TSFRESH_FEATURES, key=getKey)

            for nid, nname in sorted_feature_names:
                new_entry = None
                if int(nid) > max_known_id:
                    new_entry = '    [%s, %s]' % (str(nnid), str(nname))
                    fail_msg = '''

###    NOTICE    ###

If you adding/testing new features, you can create a new baseline from: %s

Added the new baseline (Python 2 and 3 baselines ARE different) as either:
Local path: %s/tsfresh-<NEW_VERSION>.py{2,3}.data.json.features.transposed.csv
Repo path: tests/baseline/tsfresh-<NEW_VERSION>.py{2,3}.data.json.features.transposed.csv

In the tsfresh_feature_names.py change the following:
Local path: %s/tsfresh_feature_names.py
Repo path: tests/baseline/tsfresh_feature_names.py

- TSFRESH_BASELINE_VERSION to your <NEW_VERSION>
- Add the new feature/s name to TSFRESH_FEATURES with a incremented id value

New entry for tsfresh_feature_names.py:
%s
''' % (t_fname_out_fail, baseline_dir, baseline_dir, str(new_entry))
                    if new_entry:
                        shutil.move(t_fname_out, t_fname_out_fail)
                    self.assertEqual(new_entry, None, msg=fail_msg)

                for oid, oname in sorted_feature_names:
                    if int(oid) == int(nid):
                        if str(oname) != str(nname):
                            soid = str(int(oid))
                            snid = str(int(nid))
                            soname = str(oname)
                            snname = str(oname)
                            fail_msg = '''

###    ERROR    ###

A baseline feature name for id %s has changed:
    [%s, %s]
Calculated feature name for id %s was:
    [%s, %s]

If you adding/testing new features, you can create a new baseline from: %s

Added the new baseline (Python 2 and 3 baselines ARE different) as either:
Local path: %s/tsfresh-<NEW_VERSION>.py{2,3}.data.json.features.transposed.csv
Repo path: tests/baseline/tsfresh-<NEW_VERSION>.py{2,3}.data.json.features.transposed.csv

In the tsfresh_feature_names.py change the following:
Local path: %s/tsfresh_feature_names.py
Repo path: tests/baseline/tsfresh_feature_names.py

- TSFRESH_BASELINE_VERSION to your <NEW_VERSION>
- Add the new feature/s name to TSFRESH_FEATURES with a incremented id value

''' % (soid, soid, soname, soid, soid, snname, t_fname_out, baseline_dir, baseline_dir)
                            if soname != snname:
                                shutil.move(t_fname_out, t_fname_out_fail)
                            self.assertEqual(soname, snname, msg=fail_msg)
            feature_names_match = True
        else:
            feature_names_match = False
            shutil.move(t_fname_out, t_fname_out_fail)

        self.assertEqual(feature_names_match, True)
        return True

    def test_tsfresh_baseline_json(self):
        """
        This test compares the calculated feature names AND values in the
        transposed csv to the feature names AND values in the baseline
        transposed csv.  It outputs the differences if the test fails.
        """
        start = self.test_tsfresh_baseline()
        self.assertEqual(start, True)

        if os.path.isfile(t_fname_out):
            file_created = True
        else:
            file_created = False
        self.assertEqual(file_created, True)

        df = pd.read_csv(
            t_fname_out, delimiter=',', header=0,
            names=['feature_name', 'value'])
        baseline_df = pd.read_csv(
            baseline_ts_json_baseline, delimiter=',', header=0,
            names=['feature_name', 'value'])

        # There is a more friendly user output than DataFrames comparison
        # dataframes_equal = df.equals(baseline_df)
        dataframes_equal = True
        if not dataframes_equal:
            df1 = df
            df2 = baseline_df
            ne = (df1 != df2).any(1)
            ne_stacked = (df1 != df2).stack()
            changed = ne_stacked[ne_stacked]
            changed.index.names = ['id', 'col']
            difference_locations = np.where(df1 != df2)
            changed_from = df1.values[difference_locations]
            changed_to = df2.values[difference_locations]
            _fail_msg = pd.DataFrame(
                {'from': changed_from, 'to': changed_to}, index=changed.index)
            fail_msg = 'Baseline comparison failed - %s' % _fail_msg
            self.assertEqual(dataframes_equal, True, msg=fail_msg)
        # self.assertEqual(dataframes_equal, True)

        # This is more friendly user output than the above DataFrames comparison
        calculated_features = []
        sortedlist_calcf = []
        with open(t_fname_out, 'rt') as fr:
            reader = csv.reader(fr, delimiter=',')
            sortedlist_calcf = sorted(reader, key=lambda row: row[0], reverse=True)

        for i, line in enumerate(sortedlist_calcf):
            calculated_features.append([str(line[0]), str(line[1])])

        baseline_features = []
        sortedlist_baseline = []
        with open(baseline_ts_json_baseline, 'rt') as fr:
            reader = csv.reader(fr, delimiter=',')
            sortedlist_baseline = sorted(reader, key=lambda row: row[0], reverse=True)

        for i, line in enumerate(sortedlist_baseline):
            baseline_features.append([str(line[0]), str(line[1])])

        features_equal = False
        fail_msg = 'none'
        if baseline_features == calculated_features:
            features_equal = True
        else:
            not_in_calculated = [x for x in baseline_features if x not in calculated_features]
            not_in_baseline = [x for x in calculated_features if x not in baseline_features]
            fail_msg = '''

###    NOTICE    ###

If you adding/testing new features, you can create a new baseline from: %s

Added the new baseline (Python 2 and 3 baselines ARE different) as either:
Local path: %s/tsfresh-<NEW_VERSION>.py{2,3}.data.json.features.transposed.csv
Repo path: tests/baseline/tsfresh-<NEW_VERSION>.py{2,3}.data.json.features.transposed.csv

In the tsfresh_feature_names.py change the following:
Local path: %s/tsfresh_feature_names.py
Repo path: tests/baseline/tsfresh_feature_names.py

- TSFRESH_BASELINE_VERSION to your <NEW_VERSION>
- Add the new feature/s name to TSFRESH_FEATURES with a incremented id value

NOT in baseline   :: %s

NOT in calculated :: %s''' % (t_fname_out_fail, baseline_dir, baseline_dir, str(not_in_baseline), str(not_in_calculated))
        if not features_equal:
            shutil.move(t_fname_out, t_fname_out_fail)
        self.assertEqual(features_equal, True, msg=fail_msg)

if __name__ == '__main__':
    unittest.main()
