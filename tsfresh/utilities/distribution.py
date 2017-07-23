import math

import itertools
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm


def function_with_partly_reduce(chunk_list, map_function):
    results = (map_function(chunk) for chunk in chunk_list)
    results = list(itertools.chain.from_iterable(results))
    return results


class Distributor:
    def __init__(self, n_workers, disable_progressbar, progressbar_title):
        self.n_workers = n_workers or 1
        self.disable_progressbar = disable_progressbar
        self.progressbar_title = progressbar_title

    @staticmethod
    def partition(data, chunk_size):
        # Create a generator out of the input list
        iterable = iter(data)
        while True:
            next_chunk = list(itertools.islice(iterable, chunk_size))
            if not next_chunk:
                return

            yield next_chunk

    def _calculate_best_chunksize(self, data_length):
        """
        Helper function to calculate the best chunksize for a given number of elements to calculate.

        The formula is more or less an empirical result.
        :param iterable_list_len: A length which defines how many calculations there need to be.
        :return: The chunksize which should be used.

        TODO: Investigate which is the best chunk size for different settings.
        """
        chunksize, extra = divmod(data_length, self.n_workers * 5)
        if extra:
            chunksize += 1
        return chunksize

    def map_reduce(self, map_function, data, chunk_size=None, data_length=None):
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

        result = self.reduce(result)
        result = list(itertools.chain.from_iterable(result))

        return result

    def distribute(self, func, partitioned_chunks):
        raise NotImplementedError

    def reduce(self, result):
        return result


class MapDistributor(Distributor):
    def distribute(self, func, partitioned_chunks):
        return map(func, partitioned_chunks)


class LocalDaskDistributor(Distributor):
    def __init__(self, n_workers, disable_progressbar, progressbar_title):
        Distributor.__init__(self, n_workers, disable_progressbar, progressbar_title)

        from distributed import LocalCluster, Client
        cluster = LocalCluster(n_workers=self.n_workers, processes=True)
        self.client = Client(cluster)

    def distribute(self, func, partitioned_chunks):
        result = self.client.gather(self.client.map(func, partitioned_chunks))
        return result


class MultiprocessingDistributor(Distributor):
    def __init__(self, n_workers, disable_progressbar, progressbar_title):
        Distributor.__init__(self, n_workers, disable_progressbar, progressbar_title)

        self.pool = Pool(processes=self.n_workers)

    def distribute(self, func, partitioned_chunks):
        return self.pool.imap_unordered(func, partitioned_chunks)