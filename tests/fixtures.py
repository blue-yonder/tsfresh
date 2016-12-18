from unittest import TestCase

import numpy as np
import pandas as pd


# todo: add test cases for float data
# todo: add test cases for nans, -infs, infs
# todo: add test cases with time series of length one
class DataTestCase(TestCase):
    def create_test_data_sample(self):
        cid = np.repeat([10, 500], 40)
        ckind = np.repeat(["a", "b", "a", "b"], 20)
        csort = [30, 53, 26, 35, 42, 25, 17, 67, 20, 68, 46, 12, 0, 74, 66, 31, 32,
                 2, 55, 59, 56, 60, 34, 69, 47, 15, 49, 8, 50, 73, 23, 62, 24, 33,
                 22, 70, 3, 38, 28, 75, 39, 36, 64, 13, 72, 52, 40, 16, 58, 29, 63,
                 79, 61, 78, 1, 10, 4, 6, 65, 44, 54, 48, 11, 14, 19, 43, 76, 7,
                 51, 9, 27, 21, 5, 71, 57, 77, 41, 18, 45, 37]
        cval = [11, 9, 67, 45, 30, 58, 62, 19, 56, 29, 0, 27, 36, 43, 33, 2, 24,
                71, 41, 28, 50, 40, 39, 7, 53, 23, 16, 37, 66, 38, 6, 47, 3, 61,
                44, 42, 78, 31, 21, 55, 15, 35, 25, 32, 69, 65, 70, 64, 51, 46, 5,
                77, 26, 73, 76, 75, 72, 74, 10, 57, 4, 14, 68, 22, 18, 52, 54, 60,
                79, 12, 49, 63, 8, 59, 1, 13, 20, 17, 48, 34]
        df = pd.DataFrame({"id": cid, "kind": ckind, "sort": csort, "val": cval})
        return df.set_index("id", drop=False)


    def create_one_valued_time_series(self):
        cid = [1, 2, 2]
        ckind = ["a", "a", "a"]
        csort = [1, 1, 2]
        cval = [1.0, 5.0, 6.0]
        df = pd.DataFrame({"id": cid, "kind": ckind, "sort": csort, "val": cval})
        return df


    def create_test_data_sample_with_target(self):
        """
        Small test data set with target.
        :return: timeseries df
        :return: target y which is the mean of each sample's timeseries
        """
        cid = np.repeat(range(50), 3)
        csort = list(range(3)) * 50
        cval = [1, 2, 3] * 30 + [4, 5, 6] * 20
        df = pd.DataFrame({'id': cid, 'kind': 'a', 'sort': csort, 'val': cval})
        y = pd.Series([2] * 30 + [5] * 20)
        return df, y
