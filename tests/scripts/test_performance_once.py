import os
from timeit import default_timer as timer
from ast import literal_eval

import numpy as np
import pandas as pd
import traceback
import requests

data_source = 'https://raw.githubusercontent.com/earthgecko/skyline/795a22c3a09f3bc8487d9cc7ff7a5401da1cb217/tests/baseline/stats.statsd.bad_lines_seen.20161110.csv'
multiple_timeseries = 100
savings = []
fc_times = []
once_times = []


def create_x():

    path = '/tmp/skyline/stats.statsd.bad_lines_seen'
    tmp_csv = '%s/stats.statsd.bad_lines_seen.20161110.csv' % path

    if not os.path.isfile(tmp_csv):
        print('Getting data source - %s' % data_source)
        r = None
        http_status_code = 0
        try:
            r = requests.get(data_source, timeout=10)
            http_status_code = r.status_code
            if int(http_status_code) == 200:
                print('Got data source')
        except:
            print(traceback.format_exc())
            print('error :: could not retrieve %s' % data_source)

        try:
            if not os.path.isdir(path):
                print('Need to create dir - %s' % path)
                try:
                    print('Making dir - %s' % path)
                    os.makedirs(path, mode=0o755)
                    return True
                # Python >2.5
                except OSError as exc:
                    if exc.errno == errno.EEXIST and os.path.isdir(path):
                        pass
                    else:
                        raise
            else:
                print('dir exists - %s' % path)
        except:
            print(traceback.format_exc())
            print('error :: could not create %s' % path)

        with open(tmp_csv, 'w') as fh:
            print('Creating - %s' % tmp_csv)
            fh.write(str(r.text))
    else:
        print('Using existing - %s' % tmp_csv)

    ts_count = 0
    start_df = timer()
    df = pd.DataFrame()
    # df.columns = ['metric', 'timestamp', 'value']
    df_new = pd.read_csv(tmp_csv, delimiter=',', header=None, names=['metric', 'timestamp', 'value'])
    df_new.columns = ['metric', 'timestamp', 'value']
    while ts_count < multiple_timeseries:
        df_append = df.append(df_new)
        df = df_append
        ts_count += 1
    end_df = timer()
    df_time = end_df - start_df
    x = df['value']
    print('Time to create x df of %s length :: %.6f seconds' % (str(len(x)), df_time))
    return x


def once_saving(x, func_object, func_name, times_func_used):
    """
    This method calculates time taken to calculate np values multiple times

    :param func: A numpy function or other function object
    :param func_name: The numpy function as a string
    :param times_func_used: The numpy function as a string

    :return: A list with the length of all sub-sequences where the array is either True or False. If no ones or Trues

    contained, a the list [0] is returned.
    """

    start_one = timer()
    func_result = func_object(x)
    end_one = timer()
    once_time = end_one - start_one

    # Calculate features_calculator.py time
    start = timer()
    count = 0
    while count != times_func_used:
        count += 1
        func_result = func_object(x)
    end = timer()
    x_len = len(x)
    fc_time = end - start
    once_saves = fc_time - once_time
    print('\n####\n# %s analysis\n' % (func_name))
    print(
        'Time to calculate %s once for a timeseries with %s data points :: %.6f seconds' %
        (func_name, str(x_len), once_time))
    print(
        'Time to calculate %s %s times for a timeseries with %s data points :: %.6f seconds' %
        (func_name, str(count), str(x_len), fc_time))
    print(
        'Calculating %s once is saves :: %.6f seconds or %.6f milliseconds' %
        (func_name, once_saves, (once_saves * 1000)))
    if once_time == 0:
        once_time = 0.000001
    if fc_time == 0:
        fc_time = 0.000001
    a = np.array([float(fc_time), float(once_time)], dtype=float)
    diff = np.diff(a) / np.abs(a[:-1]) * 100.
    print(
        'It is %s percent less efficient to calculate %s %s times' %
        (str(diff), func_name, str(times_func_used)))
    print('Timeseries values sum :: %s' % (str(sum(x))))

    return once_time, once_saves, fc_time


def append_times(once_time, saves, fc_time):
    savings.append(saves)
    fc_times.append(fc_time)
    once_times.append(once_time)


def calc_saving(x, func_object, func_name, fc_times):
    once_time, saves, fc_time = once_saving(x, func_object, func_name, fc_times)
    append_times(once_time, saves, fc_time)

x = create_x()

calc_saving(x, np.std, 'np.std', 5)
# ≈ (0.07 to 0.2) × average length of a human blink of an eye ( 100 to 400 ms )
# ≈ time for a nerve impulse to travel the length of a human ( 1 average human heights/maximum speed of a nerve impulse )

calc_saving(x, np.var, 'np.var', 3)
calc_saving(x, len, 'len', 26)
calc_saving(x, np.mean, 'np.mean', 13)
calc_saving(x, np.asarray, 'np.asarray', 9)
# I could not get pd.Series or np.diff to carry through
# once_time, saves, fc_time = once_saving(pd.Series, 'pd.Series', 11)
# once_time, saves, fc_time = once_saving(np.diff, 'np.diff', 4)
calc_saving(x, max, 'max', 8)
calc_saving(x, max, 'min', 8)

print('\n####\n# Overall analysis\n')
print 'Total master features_calculator.py time    :: %.6f seconds' % sum(fc_times)
print 'Total performance_once method time          :: %.6f seconds' % sum(once_times)
print 'Total time saved by performance_once method :: %.6f seconds' % sum(savings)
a = np.array([float(sum(fc_times)), float(sum(once_times))], dtype=float)
diff = np.diff(a) / np.abs(a[:-1]) * 100.
print(
    'It is %s percent less efficient to calculate values multiple times' %
    (str(diff)))

