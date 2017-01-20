from sys import version_info
from os import path
import warnings

import traceback
import numpy as np
import pandas as pd
from timeit import default_timer as timer

from tsfresh.feature_extraction import FeatureExtractionSettings
from tsfresh.utilities.profiling import start_profiling, end_profiling

from tsfresh import extract_features

warnings.simplefilter("ignore")
python_version = int(version_info[0])

fname_out = '/tmp/test_performance_varied_ts_len.timeseries.csv'
debug_log = '/tmp/tsfresh.debug.log'


def reset_log():
    if path.isfile(debug_log):
        with open(debug_log, 'w') as fh:
            pass


def dt2ut(dt):
    epoch = pd.to_datetime('1970-01-01')
    return (dt - epoch).total_seconds()


def create_df(period):

    start = timer()
    rng = pd.date_range(1262304000, periods=period, freq='T', name='timestamp_obj')
    ts = pd.Series(np.random.randn(len(rng)), rng)
    df_ts = ts.to_frame(name=None)
    df_ts.insert(0, 'id', 'test')
    df_ts.reset_index
    df_ts.columns = ['id', 'value']
    df_ts['timestamp'] = df_ts.index
    df_ts.columns = ['metric', 'value', 'timestamp']

    df_ts.reset_index
    df_ts = df_ts[['metric', 'timestamp', 'value']]
    df_ts['timestamp'] = df_ts['timestamp'].apply(dt2ut).astype(int)

    df_ts.to_csv(fname_out, index=False, header=False)
    end = timer()
    create_time = end - start
    print(
        '\nTime to create a timeseries with %s data points :: %.6f seconds' %
        (str(period), create_time))

    return fname_out


def calc_runtime(period):
    """
    This method calculates time taken to extract features on a timeseries of
    period length

    :param df: a pandas Series
    :param period: the number of samples in the timeseries

    :return: A list with the length of all sub-sequences where the array is either True or False. If no ones or Trues

    contained, a the list [0] is returned.
    """

    start = timer()
    try:
        fname_out = create_df(period)
    except:
        print(traceback.format_exc())
        print('error :: could not read buf')

    df = None

    try:
        df = pd.read_csv(fname_out, delimiter=',', header=None, names=['metric', 'timestamp', 'value'])
    except:
        print(traceback.format_exc())
        print('error :: could not created df from buf')

    df.columns = ['metric', 'timestamp', 'value']
    # profiler = start_profiling()
    df_features = extract_features(df, column_id='metric', column_sort='timestamp', column_kind=None, column_value=None)
    # profiler_fname = '/tmp/test.%s.txt' % str(period)
    # end_profiling(profiler, profiler_fname, 'cumulative')

    end = timer()
    calc_time = end - start

    print(
        'Time to calculate features for a timeseries with %s data points :: %.6f seconds\n' %
        (str(period), calc_time))

    # Analyse log
    with open(debug_log, 'r') as f:
        loglines = f.readlines()

    with open(debug_log, 'r') as f:
        loglines = []
        for i, line in enumerate(f):
            if i != 0:
                raw_line = line.rstrip('\n')
                new_line = raw_line.replace(',', ':', 1)
                metric = new_line.split(':')[2]
                try:
                    timing = float(new_line.split(':')[-1])
                    loglines.append([metric, timing])
                except:
                    pass

    all_timings = []
    unique_metrics = []
    for i, line in enumerate(loglines):
        if not line[0] in unique_metrics:
            unique_metrics.append(line[0])
    unique_metrics.sort()
    for metric in unique_metrics:
        metric_timings = []
        for i, line in enumerate(loglines):
            if line[0] == metric:
                metric_timings.append(float(line[1]))
                all_timings.append(float(line[1]))
        total_time = sum(metric_timings)
        number_of_times = str(len(metric_timings))
        print(
            '%s was calculated %s times which took :: %.6f seconds' %
            (metric, number_of_times, total_time))

    total_times = sum(all_timings)
    print(
        'Total function times took :: %.6f seconds' %
        (total_times))
    times_by = [10, 100, 1000, 10000, 100000, 10000000]
    for multiple_by in times_by:
        time_ts = total_times * multiple_by
        print(
            'Total function times for %s timeseries would take :: %.6f seconds' %
            (str(multiple_by), time_ts))

    reset_log()

    return

calc_runtime(100)
calc_runtime(1000)
calc_runtime(5000)
calc_runtime(10000)  # with logging bad
# calc_runtime(100000)  # Generates Memory Error:

example_output = '''
EXAMPLE OUTPUT:

Time to create a timeseries with 100 data points :: 0.067730 seconds
Time to calculate features for a timeseries with 100 data points :: 0.474571 seconds

Time to create a timeseries with 1000 data points :: 0.169761 seconds
Time to calculate features for a timeseries with 1000 data points :: 3.215702 seconds

Time to create a timeseries with 10000 data points :: 1.358162 seconds
Time to calculate features for a timeseries with 10000 data points :: 219.245369 seconds
'''
