# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2017
"""
This module contains the Distributor class, such objects are used to distribute the calculation of features.
Essentially, a Distributor organizes the application of feature calculators to data chunks.

Design of this module by Nils Braun
"""

import math
import itertools
import warnings
from collections import Iterable
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


def _function_with_partly_reduce(chunk_list, map_function, kwargs):
    """
    Small helper function to call a function (map_function)
    on a list of data chunks (chunk_list) and convert the results into
    a flattened list.

    This function is used to send chunks of data with a size larger than 1 to
    the workers in parallel and process these on the worker.

    :param chunk_list: A list of data chunks to process.
    :type chunk_list: list
    :param map_function: A function, which is called on each chunk in the list separately.
    :type map_function: callable

    :return: A list of the results of the function evaluated on each chunk and flattened.
    :rtype: list
    """
    kwargs = kwargs or {}
    results = (map_function(chunk, **kwargs) for chunk in chunk_list)
    results = list(itertools.chain.from_iterable(results))
    return results


def initialize_warnings_in_workers(show_warnings):
    """
    Small helper function to initialize warnings module in multiprocessing workers.

    On Windows, Python spawns fresh processes which do not inherit from warnings
    state, so warnings must be enabled/disabled before running computations.

    :param show_warnings: whether to show warnings or not.
    :type show_warnings: bool
    """
    warnings.catch_warnings()
    if not show_warnings:
        warnings.simplefilter("ignore")
    else:
        warnings.simplefilter("default")


class DistributorBaseClass:
    """
    The distributor abstract base class.

    The main purpose of the instances of the DistributorBaseClass subclasses is to evaluate a function
    (called map_function) on a list of data items (called data).

    This is done on chunks of the data, meaning, that the DistributorBaseClass classes will chunk the data into chunks,
    distribute the data and apply the feature calculator functions from
    :mod:`tsfresh.feature_extraction.feature_calculators` on the time series.

    Dependent on the implementation of the distribute function, this is done in parallel or using a cluster of nodes.
    """

    @staticmethod
    def partition(data, chunk_size):
        """
        This generator chunks a list of data into slices of length chunk_size. If the chunk_size is not a divider of the
        data length, the last slice will be shorter than chunk_size.

        :param data: The data to chunk.
        :type data: list
        :param chunk_size: Each chunks size. The last chunk may be smaller.
        :type chunk_size: int

        :return: A generator producing the chunks of data.
        :rtype: generator
        """

        iterable = iter(data)
        while True:
            next_chunk = list(itertools.islice(iterable, chunk_size))
            if not next_chunk:
                return

            yield next_chunk

    def __init__(self):
        """
        Constructs the DistributorBaseClass class
        """
        raise NotImplementedError

    def calculate_best_chunk_size(self, data_length):
        """
        Calculates the best chunk size for a list of length data_length. The current implemented formula is more or
        less an empirical result for multiprocessing case on one machine.

        :param data_length: A length which defines how many calculations there need to be.
        :type data_length: int
        :return: the calculated chunk size
        :rtype: int

        TODO: Investigate which is the best chunk size for different settings.
        """
        chunk_size, extra = divmod(data_length, self.n_workers * 5)
        if extra:
            chunk_size += 1
        return chunk_size

    def map_reduce(self, map_function, data, function_kwargs=None, chunk_size=None, data_length=None):
        """
        This method contains the core functionality of the DistributorBaseClass class.

        It maps the map_function to each element of the data and reduces the results to return a flattened list.

        How the jobs are calculated, is determined by the classes
        :func:`tsfresh.utilities.distribution.DistributorBaseClass.distribute` method,
        which can distribute the jobs in multiple threads, across multiple processing units etc.

        To not transport each element of the data individually, the data is split into chunks, according to the chunk
        size (or an empirical guess if none is given). By this, worker processes not tiny but adequate sized parts of
        the data.

        :param map_function: a function to apply to each data item.
        :type map_function: callable
        :param data: the data to use in the calculation
        :type data: iterable
        :param function_kwargs: parameters for the map function
        :type function_kwargs: dict of string to parameter
        :param chunk_size: If given, chunk the data according to this size. If not given, use an empirical value.
        :type chunk_size: int
        :param data_length: If the data is a generator, you have to set the length here. If it is none, the
          length is deduced from the len of the data.
        :type data_length: int

        :return: the calculated results
        :rtype: list
        """
        if data_length is None:
            data_length = len(data)

        if not chunk_size:
            chunk_size = self.calculate_best_chunk_size(data_length)

        chunk_generator = self.partition(data, chunk_size=chunk_size)

        map_kwargs = {"map_function": map_function, "kwargs": function_kwargs}

        if hasattr(self, "progressbar_title"):
            total_number_of_expected_results = math.ceil(data_length / chunk_size)
            result = tqdm(self.distribute(_function_with_partly_reduce, chunk_generator, map_kwargs),
                          total=total_number_of_expected_results,
                          desc=self.progressbar_title, disable=self.disable_progressbar)
        else:
            result = self.distribute(_function_with_partly_reduce, chunk_generator, map_kwargs),

        result = list(itertools.chain.from_iterable(result))

        return result

    def distribute(self, func, partitioned_chunks, kwargs):
        """
        This abstract base function distributes the work among workers, which can be threads or nodes in a cluster.
        Must be implemented in the derived classes.

        :param func: the function to send to each worker.
        :type func: callable
        :param partitioned_chunks: The list of data chunks - each element is again
            a list of chunks - and should be processed by one worker.
        :type partitioned_chunks: iterable
        :param kwargs: parameters for the map function
        :type kwargs: dict of string to parameter

        :return: The result of the calculation as a list - each item should be the result of the application of func
            to a single element.
        """
        raise NotImplementedError

    def close(self):
        """
        Abstract base function to clean the DistributorBaseClass after use, e.g. close the connection to a DaskScheduler
        """
        pass


class MapDistributor(DistributorBaseClass):
    """
    Distributor using the python build-in map, which calculates each job sequentially one after the other.
    """

    def __init__(self, disable_progressbar=False, progressbar_title="Feature Extraction"):
        """
        Creates a new MapDistributor instance

        :param disable_progressbar: whether to show a progressbar or not.
        :type disable_progressbar: bool
        :param progressbar_title: the title of the progressbar
        :type progressbar_title: basestring
        """
        self.disable_progressbar = disable_progressbar
        self.progressbar_title = progressbar_title

    def distribute(self, func, partitioned_chunks, kwargs):
        """
        Calculates the features in a sequential fashion by pythons map command

        :param func: the function to send to each worker.
        :type func: callable
        :param partitioned_chunks: The list of data chunks - each element is again
            a list of chunks - and should be processed by one worker.
        :type partitioned_chunks: iterable
        :param kwargs: parameters for the map function
        :type kwargs: dict of string to parameter

        :return: The result of the calculation as a list - each item should be the result of the application of func
            to a single element.
        """
        return map(partial(func, **kwargs), partitioned_chunks)

    def calculate_best_chunk_size(self, data_length):
        """
        For the map command, which calculates the features sequentially, a the chunk_size of 1 will be used.

        :param data_length: A length which defines how many calculations there need to be.
        :type data_length: int
        """
        return 1


class LocalDaskDistributor(DistributorBaseClass):
    """
    Distributor using a local dask cluster and inproc communication.
    """

    def __init__(self, n_workers):
        """

        Initiates a LocalDaskDistributor instance.

        :param n_workers: How many workers should the local dask cluster have?
        :type n_workers: int
        """

        from distributed import LocalCluster, Client
        import tempfile

        # attribute .local_dir_ is the path where the local dask workers store temporary files
        self.local_dir_ = tempfile.mkdtemp()
        cluster = LocalCluster(n_workers=n_workers, processes=False, local_dir=self.local_dir_)

        self.client = Client(cluster)
        self.n_workers = n_workers

    def distribute(self, func, partitioned_chunks, kwargs):
        """
        Calculates the features in a parallel fashion by distributing the map command to the dask workers on a local
        machine

        :param func: the function to send to each worker.
        :type func: callable
        :param partitioned_chunks: The list of data chunks - each element is again
            a list of chunks - and should be processed by one worker.
        :type partitioned_chunks: iterable
        :param kwargs: parameters for the map function
        :type kwargs: dict of string to parameter

        :return: The result of the calculation as a list - each item should be the result of the application of func
            to a single element.
        """

        if isinstance(partitioned_chunks, Iterable):
            # since dask 2.0.0 client map no longer accepts iterables
            partitioned_chunks = list(partitioned_chunks)
        result = self.client.gather(self.client.map(partial(func, **kwargs), partitioned_chunks))
        return [item for sublist in result for item in sublist]

    def close(self):
        """
        Closes the connection to the local Dask Scheduler
        """
        self.client.close()


class ClusterDaskDistributor(DistributorBaseClass):
    """
    Distributor using a dask cluster, meaning that the calculation is spread over a cluster
    """

    def __init__(self, address):
        """
        Sets up a distributor that connects to a Dask Scheduler to distribute the calculaton of the features

        :param address: the ip address and port number of the Dask Scheduler
        :type address: str
        """

        from distributed import Client

        self.client = Client(address=address)

    def calculate_best_chunk_size(self, data_length):
        """
        Uses the number of dask workers in the cluster (during execution time, meaning when you start the extraction)
        to find the optimal chunk_size.

        :param data_length: A length which defines how many calculations there need to be.
        :type data_length: int
        """
        n_workers = len(self.client.scheduler_info()["workers"])
        chunk_size, extra = divmod(data_length, n_workers * 5)
        if extra:
            chunk_size += 1
        return chunk_size

    def distribute(self, func, partitioned_chunks, kwargs):
        """
        Calculates the features in a parallel fashion by distributing the map command to the dask workers on a cluster

        :param func: the function to send to each worker.
        :type func: callable
        :param partitioned_chunks: The list of data chunks - each element is again
            a list of chunks - and should be processed by one worker.
        :type partitioned_chunks: iterable
        :param kwargs: parameters for the map function
        :type kwargs: dict of string to parameter

        :return: The result of the calculation as a list - each item should be the result of the application of func
            to a single element.
        """
        if isinstance(partitioned_chunks, Iterable):
            # since dask 2.0.0 client map no longer accepts iterables
            partitioned_chunks = list(partitioned_chunks)
        result = self.client.gather(self.client.map(partial(func, **kwargs), partitioned_chunks))
        return [item for sublist in result for item in sublist]

    def close(self):
        """
        Closes the connection to the Dask Scheduler
        """
        self.client.close()


class MultiprocessingDistributor(DistributorBaseClass):
    """
    Distributor using a multiprocessing Pool to calculate the jobs in parallel on the local machine.
    """

    def __init__(self, n_workers, disable_progressbar=False, progressbar_title="Feature Extraction",
                 show_warnings=True):
        """
        Creates a new MultiprocessingDistributor instance

        :param n_workers: How many workers should the multiprocessing pool have?
        :type n_workers: int
        :param disable_progressbar: whether to show a progressbar or not.
        :type disable_progressbar: bool
        :param progressbar_title: the title of the progressbar
        :type progressbar_title: basestring
        :param show_warnings: whether to show warnings or not.
        :type show_warnings: bool
        """
        self.pool = Pool(processes=n_workers, initializer=initialize_warnings_in_workers, initargs=(show_warnings,))
        self.n_workers = n_workers
        self.disable_progressbar = disable_progressbar
        self.progressbar_title = progressbar_title

    def distribute(self, func, partitioned_chunks, kwargs):
        """
        Calculates the features in a parallel fashion by distributing the map command to a thread pool

        :param func: the function to send to each worker.
        :type func: callable
        :param partitioned_chunks: The list of data chunks - each element is again
            a list of chunks - and should be processed by one worker.
        :type partitioned_chunks: iterable
        :param kwargs: parameters for the map function
        :type kwargs: dict of string to parameter

        :return: The result of the calculation as a list - each item should be the result of the application of func
            to a single element.
        """
        return self.pool.imap_unordered(partial(func, **kwargs), partitioned_chunks)

    def close(self):
        """
        Collects the result from the workers and closes the thread pool.
        """
        self.pool.close()
        self.pool.terminate()
        self.pool.join()
