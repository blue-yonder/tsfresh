# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
import math

import itertools
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm


def function_with_partly_reduce(chunk_list, map_function):
    """
    Small helper function to call a function (map_function)
    on a list of data chunks (chunk_list) and convert the results into
    a flattened list.
    This function is used to send chunks of data with a size larger than 1 to
    the workers in parallel and process these on the worker.

    :param chunk_list: A list of data chunks to process.
    :type chunk_list: list
    :param map_function: A function, which is called on each chunk in the list seperately.
    :type map_function: callable
    :return: A list of the results of the function evaluated on each chunk and flattened.
    :rtype: list
    """
    results = (map_function(chunk) for chunk in chunk_list)
    results = list(itertools.chain.from_iterable(results))
    return results


class Distributor:
    """
    Base class for each distributors. Instances of this type can evaluate a function
    (called map_function) in a list of data items (called data) - chunked automatically
    according to chunk_size - and combine the results.
    Dependent on the implementation of the distribute function, this is done in parallel,
    using a cluster of nodes etc.
    """
    def __init__(self, n_workers, disable_progressbar, progressbar_title):
        """
        Create a new instance of the distributor.
        :param n_workers: How many workers should the distributor have. How this information is used
            depends on the implementation of the given distributor.
        :type n_workers: int
        :param disable_progressbar: Show a progressbar or not.
        :type disable_progressbar: bool
        :param progressbar_title: Which title should the progressbar have.
        :type progressbar_title: basestring
        """
        self.n_workers = n_workers or 1
        self.disable_progressbar = disable_progressbar
        self.progressbar_title = progressbar_title

    @staticmethod
    def partition(data, chunk_size):
        """
        Helper function to chunk a list of data items with the given chunk size.
        This is done with the help of some iterator tools.
        :param data: The data to chunk.
        :type data: list
        :param chunk_size: The size of one chunk. The last chunk may be smaller.
        :type chunk_size: int
        :return: A generator producing the chunks of data.
        :rtype: generator
        """
        # Create a generator out of the input list
        iterable = iter(data)
        while True:
            next_chunk = list(itertools.islice(iterable, chunk_size))
            if not next_chunk:
                return

            yield next_chunk

    def _calculate_best_chunksize(self, data_length):
        """
        Helper function to calculate the best chunk size for a given number of elements to calculate.

        The formula is more or less an empirical result.
        :param data_length: A length which defines how many calculations there need to be.
        :type data_length: int
        :return: The chunk size which should be used.
        :rtype: int

        TODO: Investigate which is the best chunk size for different settings.
        """
        chunksize, extra = divmod(data_length, self.n_workers * 5)
        if extra:
            chunksize += 1
        return chunksize

    def map_reduce(self, map_function, data, chunk_size=None, data_length=None):
        """
        Main function of the class: calculate the map_function for each element in the data and return
        the flattened list of results.

        How the jobs are calculated, is determined by the distribute function in the class,
        which can e.g. distribute the jobs in multiple processes etc.

        To save streaming capabilities, the data is chunked according to the chunk size (or an empirical
        guess if none is given) and each worker processes a larger part of the data.

        :param map_function: The function to calculate for each data item.
        :type map_function: callable
        :param data: The data to use in the calculation
        :type data: iterable
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
            chunk_size = self._calculate_best_chunksize(data_length)

        partitioned_chunks = self.partition(data, chunk_size=chunk_size)

        total_number_of_expected_results = math.ceil(data_length / chunk_size)

        specialized_function_with_partly_reduce = partial(function_with_partly_reduce, map_function=map_function)

        result = tqdm(self.distribute(specialized_function_with_partly_reduce, partitioned_chunks),
                      total=total_number_of_expected_results,
                      desc=self.progressbar_title, disable=self.disable_progressbar)

        result = list(itertools.chain.from_iterable(result))

        return result

    def distribute(self, func, partitioned_chunks):
        """
        Abstract base function to do the distribution work of jobs.
        Must be implemented in the derived classes.
        :param func: The function to send to each worker.
        :type func: callable
        :param partitioned_chunks: The list of data chunks - each element is again
            a list of chunks - and should be processed by one worker.
        :type partitioned_chunks: iterable
        :return: The result of the calculation as a list - each item should be the result of a single
            worker.
        """
        raise NotImplementedError


class MapDistributor(Distributor):
    """
    Distributor using the python build-in map, which calculates each job one after the other.
    """
    def distribute(self, func, partitioned_chunks):
        return map(func, partitioned_chunks)


class LocalDaskDistributor(Distributor):
    """
    Distributor using a local dask cluster and inproc communication.
    """
    def __init__(self, n_workers, disable_progressbar, progressbar_title):
        Distributor.__init__(self, n_workers, disable_progressbar, progressbar_title)

        from distributed import LocalCluster, Client
        cluster = LocalCluster(n_workers=self.n_workers, processes=True)
        self.client = Client(cluster)

    def distribute(self, func, partitioned_chunks):
        result = self.client.gather(self.client.map(func, partitioned_chunks))
        return result


class MultiprocessingDistributor(Distributor):
    """
    Distributor using a multiprocessing Pool to calculate the jobs in parallel on the local machine.
    """
    def __init__(self, n_workers, disable_progressbar, progressbar_title):
        Distributor.__init__(self, n_workers, disable_progressbar, progressbar_title)

        self.pool = Pool(processes=self.n_workers)

    def distribute(self, func, partitioned_chunks):
        return self.pool.imap_unordered(func, partitioned_chunks)