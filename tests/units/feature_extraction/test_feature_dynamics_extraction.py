# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import os
import tempfile

import numpy as np
import pandas as pd
from mock import Mock

from fixtures import DataTestCase
from tsfresh.feature_extraction.extraction import extract_feature_dynamics

from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.utilities.distribution import IterableDistributorBaseClass, MapDistributor

from tsfresh.feature_extraction.extraction import extract_features


class DynamicsExtractionTestCase(DataTestCase):
    """The unit tests in this module make sure if the time series feature dynamics are created properly"""

    def setUp(self):
        self.n_jobs = 1
        self.directory = tempfile.gettempdir()

    def test_extract_feature_dynamics(self):
        # TODO: implement more methods and test more aspects
        df = self.create_test_data_sample()
        extracted_feature_dynamics = extract_feature_dynamics(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            window_length=15,
            n_jobs=self.n_jobs,
        )

        self.assertIsInstance(extracted_feature_dynamics, pd.DataFrame)
        self.assertTrue(
            np.all(
                extracted_feature_dynamics[
                    "a||abs_energy__variance_larger_than_standard_deviation"
                ]
                == np.array([1.0, 1.0])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics["a||sum_values__sum_values"]
                == np.array([691, 1017])
            )
        )

        self.assertTrue(
            np.all(
                extracted_feature_dynamics["b||sum_values__sum_values"]
                == np.array([757, 695])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics["b||minimum__minimum"] == np.array([3, 1])
            )
        )

        df_sts = self.create_one_valued_time_series()

        # only calculate some features as small amount of data means quantile fcs will break
        self.name_to_param = {
            "maximum": None,
            "sum_values": None,
            "abs_energy": None,
            "minimum": None,
            "mean": None,
            "median": None,
        }

        extracted_feature_dynamics_sts = extract_feature_dynamics(
            df_sts,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            window_length=3,
            n_jobs=self.n_jobs,
            feature_timeseries_fc_parameters=self.name_to_param,
            feature_dynamics_fc_parameters=self.name_to_param,
        )

        self.assertIsInstance(extracted_feature_dynamics_sts, pd.DataFrame)
        self.assertTrue(
            np.all(
                extracted_feature_dynamics_sts["a||maximum__maximum"]
                == np.array([1.0, 6.0])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics_sts["a||sum_values__sum_values"]
                == np.array([1.0, 11.0])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics_sts["a||minimum__minimum"]
                == np.array([1.0, 5.0])
            )
        )

    def test_extract_feature_dynamics_uses_only_kind_to_fc_settings(self):
        df = self.create_test_data_sample()
        extracted_features = extract_feature_dynamics(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            window_length=5,
            n_jobs=self.n_jobs,
            feature_timeseries_kind_to_fc_parameters={
                "a": {"maximum": None, "minimum": None}
            },
        )
        assert len(extracted_features) == 2

    def test_extract_feature_dynamics_for_one_time_series(self):
        # TODO: implement more methods and test more aspects
        df = self.create_test_data_sample()
        settings = ComprehensiveFCParameters()

        extracted_feature_dynamics = extract_feature_dynamics(
            df,
            feature_timeseries_fc_parameters=settings,
            feature_dynamics_fc_parameters=settings,
            column_value="val",
            column_id="id",
            column_kind="kind",
            column_sort="sort",
            window_length=5,
        )

        self.assertIsInstance(extracted_feature_dynamics, pd.DataFrame)
        self.assertTrue(
            np.all(
                extracted_feature_dynamics["b||sum_values__sum_values"]
                == np.array([757, 695])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics["b||minimum__minimum"] == np.array([3, 1])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics["b||length__mean"] == np.array([5.0, 5.0])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics["b||median__median"]
                == np.array([39.5, 28.0])
            )
        )

        df_sts = self.create_one_valued_time_series()

        # only calculate some features as small amount of data means quantile fcs will break
        self.name_to_param = {
            "maximum": None,
            "sum_values": None,
            "abs_energy": None,
            "minimum": None,
            "mean": None,
            "median": None,
        }
        extracted_feature_dynamics_sts = extract_feature_dynamics(
            df_sts,
            feature_timeseries_fc_parameters=self.name_to_param,
            feature_dynamics_fc_parameters=self.name_to_param,
            column_value="val",
            column_id="id",
            column_kind="kind",
            column_sort="sort",
            window_length=2,
        )

        self.assertIsInstance(extracted_feature_dynamics_sts, pd.DataFrame)
        self.assertTrue(
            np.all(
                extracted_feature_dynamics_sts["a||maximum__maximum"]
                == np.array([1.0, 6.0])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics_sts["a||sum_values__sum_values"]
                == np.array([1.0, 11.0])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics_sts["a||minimum__minimum"]
                == np.array([1.0, 5.0])
            )
        )

    def test_extract_feature_dynamics_for_index_based_functions(self):
        df = self.create_test_data_sample_with_time_index()

        settings = {
            "linear_trend": [{"attr": "slope"}],
        }

        extracted_feature_dynamics = extract_feature_dynamics(
            df,
            feature_timeseries_fc_parameters=settings,
            feature_dynamics_fc_parameters=settings,
            column_value="val",
            column_id="id",
            column_kind="kind",
            column_sort="sort",
            window_length=15,
        )

        self.assertIsInstance(extracted_feature_dynamics, pd.DataFrame)

        # Test that the index of the returned df is the ID and not the timestamp
        self.assertTrue(extracted_feature_dynamics.index.dtype != df.index.dtype)
        self.assertTrue(extracted_feature_dynamics.index.dtype == df["id"].dtype)
        self.assertEqual(
            sorted(extracted_feature_dynamics.index.unique().tolist()),
            sorted(df["id"].unique().tolist()),
        )

    def test_extract_feature_dynamics_after_randomisation(self):
        df = self.create_test_data_sample()
        df_random = df.copy().sample(frac=1)

        extracted_feature_dynamics = extract_feature_dynamics(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
            window_length=15,
        ).sort_index()
        extracted_feature_dynamics_from_random = extract_feature_dynamics(
            df_random,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
            window_length=15,
        ).sort_index()

        self.assertCountEqual(
            extracted_feature_dynamics.columns,
            extracted_feature_dynamics_from_random.columns,
        )

        for col in extracted_feature_dynamics:
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    extracted_feature_dynamics[col],
                    extracted_feature_dynamics_from_random[col],
                )
            )

    def test_profiling_file_written_out(self):

        df = pd.DataFrame(
            data={"id": np.repeat([1, 2], 10), "val": np.random.normal(0, 1, 20)}
        )
        profiling_filename = os.path.join(self.directory, "test_profiling.txt")
        X = extract_feature_dynamics(
            df,
            column_id="id",
            column_value="val",
            n_jobs=self.n_jobs,
            profile=True,
            profiling_filename=profiling_filename,
            window_length=5,
        )

        self.assertTrue(os.path.isfile(profiling_filename))
        os.remove(profiling_filename)

    def test_profiling_cumulative_file_written_out(self):

        PROFILING_FILENAME = os.path.join(
            self.directory, "test_profiling_cumulative.txt"
        )
        PROFILING_SORTING = "cumulative"

        df = pd.DataFrame(
            data={"id": np.repeat([1, 2], 10), "val": np.random.normal(0, 1, 20)}
        )
        extract_feature_dynamics(
            df,
            column_id="id",
            column_value="val",
            n_jobs=self.n_jobs,
            profile=True,
            profiling_filename=PROFILING_FILENAME,
            profiling_sorting=PROFILING_SORTING,
            window_length=5,
        )

        self.assertTrue(os.path.isfile(PROFILING_FILENAME))
        os.remove(PROFILING_FILENAME)

    def test_extract_feature_dynamics_without_settings(self):
        df = pd.DataFrame(
            data={
                "id": np.repeat([1, 2], 10),
                "value1": np.random.normal(0, 1, 20),
                "value2": np.random.normal(0, 1, 20),
            }
        )
        X = extract_feature_dynamics(
            df, column_id="id", n_jobs=self.n_jobs, window_length=5
        )
        self.assertIn("value1||maximum__maximum", list(X.columns))
        self.assertIn("value2||maximum__maximum", list(X.columns))

    def test_extract_feature_dynamics_with_and_without_parallelization(self):
        df = self.create_test_data_sample()

        feature_dynamics_parallel = extract_feature_dynamics(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
            window_length=15,
        )

        feature_dynamics_serial = extract_feature_dynamics(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=0,
            window_length=15,
        )

        self.assertCountEqual(
            feature_dynamics_parallel.columns, feature_dynamics_serial.columns
        )

        for col in feature_dynamics_parallel.columns:
            np.testing.assert_array_almost_equal(
                feature_dynamics_parallel[col], feature_dynamics_serial[col]
            )

    def test_extract_index_preservation(self):
        df = self.create_test_data_nearly_numerical_indices()
        extracted_feature_dynamics = extract_feature_dynamics(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
            window_length=15,
        )

        self.assertIsInstance(extracted_feature_dynamics, pd.DataFrame)
        self.assertEqual(set(df["id"]), set(extracted_feature_dynamics.index))

    def test_extract_feature_dynamics_alphabetically_sorted(self):
        df = self.create_test_data_sample()

        features = extract_feature_dynamics(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            window_length=15,
        )

        for col_name in features.columns:
            # split out the configuration of the features calculator
            col_name_chunks = col_name.split("__")
            # the name is always at the beginning, so remove it. Also remove the kind of the column
            col_name_chunks = col_name_chunks[2:]

            self.assertEqual(col_name_chunks, list(sorted(col_name_chunks)))


class ParallelDynamicsExtractionTestCase(DataTestCase):
    def setUp(self):
        self.n_jobs = 2

        # only calculate some features to reduce load on travis ci
        self.name_to_param = {
            "maximum": None,
            "sum_values": None,
            "abs_energy": None,
            "minimum": None,
            "mean": None,
            "median": None,
        }

    def test_extract_feature_dynamics(self):
        # TODO: implement more methods and test more aspects
        df = self.create_test_data_sample()
        extracted_feature_dynamics = extract_feature_dynamics(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            n_jobs=self.n_jobs,
            window_length=15,
        )

        self.assertIsInstance(extracted_feature_dynamics, pd.DataFrame)
        self.assertTrue(
            np.all(
                extracted_feature_dynamics["a||maximum__maximum"] == np.array([71, 77])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics["a||sum_values__sum_values"]
                == np.array([691, 1017])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics["b||sum_values__sum_values"]
                == np.array([757, 695])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics["b||minimum__minimum"] == np.array([3, 1])
            )
        )
        self.assertTrue(
            np.all(
                extracted_feature_dynamics["b||median__length"] == np.array([2.0, 2.0])
            )
        )


class DynamicsDistributorUsageTestCase(DataTestCase):
    def setUp(self):
        # only calculate some features to reduce load on travis ci
        self.name_to_param = {"maximum": None}

    def test_distributor_map_reduce_is_called(self):
        df = self.create_test_data_sample()

        mock = Mock(spec=IterableDistributorBaseClass)
        mock.close.return_value = None
        mock.map_reduce.return_value = []

        X = extract_feature_dynamics(
            timeseries_container=df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            feature_timeseries_fc_parameters=self.name_to_param,
            feature_dynamics_fc_parameters=self.name_to_param,
            distributor=mock,
            window_length=15,
        )

        self.assertTrue(mock.map_reduce.called)

    def test_distributor_close_is_called(self):
        df = self.create_test_data_sample()

        mock = MapDistributor()
        mock.close = Mock()
        mock.close.return_value = None

        X = extract_feature_dynamics(
            timeseries_container=df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            feature_timeseries_fc_parameters=self.name_to_param,
            feature_dynamics_fc_parameters=self.name_to_param,
            distributor=mock,
            window_length=15,
        )

        self.assertTrue(mock.close.called)

