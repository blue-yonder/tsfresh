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


class Distributor:
    """
    The distributor base class. 
    
    The main purpose of the instances of this type is to evaluate a function (called map_function) on a list of data 
    items (called data).
    
    This is done on chunks of the data, meaning, that the Distributor classes will chunk the data into chunks, 
    distribute the data and apply the function on the elements of the chunks.  
    
    Dependent on the implementation of the distribute function, this is done in parallel or using a cluster of nodes
    """

    @staticmethod
    def partition(data, chunk_size):
        """
        This generator chunks a list of data into slices of size chunk_size, the last slice can be shorter

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
        Constructs the Distributor class
        """
        raise NotImplementedError

    def _calculate_best_chunk_size(self, data_length):
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
        This method contains the core functionality of the Distributor class.
        
        It maps the map_function to each element of the data and reduces the results to return a flattened list.

        How the jobs are calculated, is determined by the classes 
        :func:`tsfresh.utilities.distribution.Distributor.distribute`method, which can distribute the jobs in multiple 
        threads, across multiple processing units etc.

        To not transport each element of the data invidually, the data is chunked, according to the chunk size (or an 
        empirical guess if none is given). By this, worker processes not tiny but adequate sized parts of the data.

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
            chunk_size = self._calculate_best_chunk_size(data_length)

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
        
        :return: The result of the calculation as a list - each item should be the result of the application of func 
            to a single element.
        """
        raise NotImplementedError

    def close(self):
        """
        Abstract base function to clean the Distributor after use, e.g. close the connection to a DaskScheduler
        """
        pass


class MapDistributor(Distributor):
    """
    Distributor using the python build-in map, which calculates each job sequentially one after the other.
    """

    def __init__(self, n_workers=1, disable_progressbar=False, progressbar_title=None):
        """
        Creates a new MapDistributor instance

        :param n_workers: How many workers should the distributor have. How this information is used
            depends on the implementation of the given distributor.
        :type n_workers: int
        :param disable_progressbar: whether to show a progressbar or not.
        :type disable_progressbar: bool
        :param progressbar_title: the title of the progressbar
        :type progressbar_title: basestring
        """

        self.n_workers = n_workers or 1
        self.disable_progressbar = disable_progressbar
        self.progressbar_title = progressbar_title
    
    def distribute(self, func, partitioned_chunks, kwargs):
        return map(partial(func, **kwargs), partitioned_chunks)

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

    def close(self):
        self.client.close()


class ClusterDaskDistributor(Distributor):
    """
    Distributor using a dask cluster, meaning that the calculation is spread over a cluster
    """
    def __init__(self, n_workers, disable_progressbar, progressbar_title, address):

        Distributor.__init__(self, n_workers, disable_progressbar, progressbar_title)

        from distributed import Client
        from tsfresh.utilities.string_manipulation import is_valid_ip_and_port

        assert is_valid_ip_and_port(address)
        self.client = Client(address=address)

    def _calculate_best_chunksize(self, data_length):
        """
        Uses the number of dask workers during execution to setup the chunksize
        """

        n_workers = len(self.client.scheduler_info()["workers"])
        chunksize, extra = divmod(data_length, n_workers * 5)
        if extra:
            chunksize += 1
        return chunksize

    def distribute(self, func, partitioned_chunks):
        result = self.client.gather(self.client.map(func, partitioned_chunks))
        return result

    def close(self):
        self.client.close()


class MultiprocessingDistributor(Distributor):
    """
    Distributor using a multiprocessing Pool to calculate the jobs in parallel on the local machine.
    """
    def __init__(self, n_workers, disable_progressbar, progressbar_title):
        Distributor.__init__(self, n_workers, disable_progressbar, progressbar_title)

        self.pool = Pool(processes=self.n_workers)

    def distribute(self, func, partitioned_chunks, kwargs):
        return self.pool.imap_unordered(partial(func, **kwargs), partitioned_chunks)

    def close(self):
        self.pool.close()
        self.pool.terminate()
        self.pool.join()