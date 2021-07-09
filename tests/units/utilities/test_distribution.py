# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import sys
from unittest import TestCase, skipIf

import numpy as np
import pandas as pd
from distributed import Client, LocalCluster

from tests.fixtures import DataTestCase
from tsfresh import extract_features
from tsfresh.utilities.distribution import (
    ClusterDaskDistributor,
    LocalDaskDistributor,
    MultiprocessingDistributor,
)


class MultiprocessingDistributorTestCase(TestCase):
    def test_partition(self):

        distributor = MultiprocessingDistributor(n_workers=1)

        data = [1, 3, 10, -10, 343.0]
        distro = distributor.partition(data, 3)
        self.assertEqual(next(distro), [1, 3, 10])
        self.assertEqual(next(distro), [-10, 343.0])

        data = np.arange(10)
        distro = distributor.partition(data, 2)
        self.assertEqual(next(distro), [0, 1])
        self.assertEqual(next(distro), [2, 3])

    def test__calculate_best_chunk_size(self):

        distributor = MultiprocessingDistributor(n_workers=2)
        self.assertEqual(distributor.calculate_best_chunk_size(10), 1)
        self.assertEqual(distributor.calculate_best_chunk_size(11), 2)
        self.assertEqual(distributor.calculate_best_chunk_size(100), 10)
        self.assertEqual(distributor.calculate_best_chunk_size(101), 11)

        distributor = MultiprocessingDistributor(n_workers=3)
        self.assertEqual(distributor.calculate_best_chunk_size(10), 1)
        self.assertEqual(distributor.calculate_best_chunk_size(30), 2)
        self.assertEqual(distributor.calculate_best_chunk_size(31), 3)


class LocalDaskDistributorTestCase(DataTestCase):
    def test_local_dask_cluster_extraction_one_worker(self):

        Distributor = LocalDaskDistributor(n_workers=1)

        df = self.create_test_data_sample()
        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            distributor=Distributor,
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(np.all(extracted_features.a__maximum == np.array([71, 77])))
        self.assertTrue(
            np.all(extracted_features.a__sum_values == np.array([691, 1017]))
        )
        self.assertTrue(
            np.all(extracted_features.a__abs_energy == np.array([32211, 63167]))
        )
        self.assertTrue(
            np.all(extracted_features.b__sum_values == np.array([757, 695]))
        )
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(
            np.all(extracted_features.b__abs_energy == np.array([36619, 35483]))
        )
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))

    def test_local_dask_cluster_extraction_two_worker(self):

        Distributor = LocalDaskDistributor(n_workers=2)

        df = self.create_test_data_sample()
        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            distributor=Distributor,
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(np.all(extracted_features.a__maximum == np.array([71, 77])))
        self.assertTrue(
            np.all(extracted_features.a__sum_values == np.array([691, 1017]))
        )
        self.assertTrue(
            np.all(extracted_features.a__abs_energy == np.array([32211, 63167]))
        )
        self.assertTrue(
            np.all(extracted_features.b__sum_values == np.array([757, 695]))
        )
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(
            np.all(extracted_features.b__abs_energy == np.array([36619, 35483]))
        )
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))


class ClusterDaskDistributorTestCase(DataTestCase):
    @skipIf(sys.version_info < (3, 6, 0), "dask requires python >= 3.6")
    def test_dask_cluster_extraction_one_worker(self):
        cluster = LocalCluster(
            n_workers=1, threads_per_worker=1, dashboard_address=None
        )
        client = Client(cluster)
        address = client.scheduler_info()["address"]
        Distributor = ClusterDaskDistributor(address=address)

        df = self.create_test_data_sample()
        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            distributor=Distributor,
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(np.all(extracted_features.a__maximum == np.array([71, 77])))
        self.assertTrue(
            np.all(extracted_features.a__sum_values == np.array([691, 1017]))
        )
        self.assertTrue(
            np.all(extracted_features.a__abs_energy == np.array([32211, 63167]))
        )
        self.assertTrue(
            np.all(extracted_features.b__sum_values == np.array([757, 695]))
        )
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(
            np.all(extracted_features.b__abs_energy == np.array([36619, 35483]))
        )
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))
        cluster.close()

    @skipIf(sys.version_info < (3, 6, 0), "dask requires python >= 3.6")
    def test_dask_cluster_extraction_two_workers(self):
        cluster = LocalCluster(
            n_workers=2, threads_per_worker=1, dashboard_address=None
        )
        client = Client(cluster)
        address = client.scheduler_info()["address"]
        Distributor = ClusterDaskDistributor(address=address)

        df = self.create_test_data_sample()
        extracted_features = extract_features(
            df,
            column_id="id",
            column_sort="sort",
            column_kind="kind",
            column_value="val",
            distributor=Distributor,
        )

        self.assertIsInstance(extracted_features, pd.DataFrame)
        self.assertTrue(np.all(extracted_features.a__maximum == np.array([71, 77])))
        self.assertTrue(
            np.all(extracted_features.a__sum_values == np.array([691, 1017]))
        )
        self.assertTrue(
            np.all(extracted_features.a__abs_energy == np.array([32211, 63167]))
        )
        self.assertTrue(
            np.all(extracted_features.b__sum_values == np.array([757, 695]))
        )
        self.assertTrue(np.all(extracted_features.b__minimum == np.array([3, 1])))
        self.assertTrue(
            np.all(extracted_features.b__abs_energy == np.array([36619, 35483]))
        )
        self.assertTrue(np.all(extracted_features.b__mean == np.array([37.85, 34.75])))
        self.assertTrue(np.all(extracted_features.b__median == np.array([39.5, 28.0])))
        cluster.close()
