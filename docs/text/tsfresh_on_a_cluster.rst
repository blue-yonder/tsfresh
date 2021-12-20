.. _tsfresh-on-a-cluster-label:

.. role:: python(code)
    :language: python

Parallelization
===============

The feature extraction, the feature selection, as well as the rolling, offer the possibility of parallelization.
By default, all of those tasks are parallelized by tsfresh.
Here we discuss the different settings to control the parallelization.
To achieve the best results for your use-case you should experiment with the parameters.

.. NOTE::
    This document describes parallelization to speed up processing time.
    If you are working with large amounts of data (which might not fit into memory),
    check :ref:`large-data-label`.

Please, let us know about your results tuning the below mentioned parameters! It will help improve the documentation as
well as the default settings.

Parallelization of Feature Selection
------------------------------------

We use a :class:`multiprocessing.Pool` to parallelize the calculation of the p-values for each feature. On
instantiation we set the Pool's number of worker processes to
`n_jobs`. This field defaults to
the number of processors on the current system. We recommend setting it to the maximum number of available (and
otherwise idle) processors.

The chunksize of the Pool's map function is another important parameter to consider. It can be set via the
`chunksize` field. By default it is up to
:class:`multiprocessing.Pool` is parallelisation parameter. One data chunk is
defined as a singular time series for one id and one kind. The chunksize is the
number of chunks that are submitted as one task to one worker process.  If you
set the chunksize to 10, then it means that one worker task corresponds to
calculate all features for 10 id/kind time series combinations. If it is set
to None, depending on distributor, heuristics are used to find the optimal
chunksize. The chunksize can have a crucial influence on the optimal cluster
performance and should be optimised in benchmarks for the problem at hand.

Parallelization of Feature Extraction
-------------------------------------

For the feature extraction tsfresh exposes the parameters
`n_jobs` and `chunksize`. Both behave similarly to the parameters
for the feature selection.

To do performance studies and profiling, it is sometimes useful to turn off parallelization. This can be
done by setting the parameter `n_jobs` to 0.

Parallelization beyond a single machine
---------------------------------------

The high volume of time series data can demand an analysis at scale.
So, time series need to be processed on a group of computational units instead of a singular machine.

Accordingly, it may be necessary to distribute the extraction of time series features to a cluster.
It is possible to extract features with *tsfresh* in a distributed fashion.
In the following paragraphs we discuss how to setup a distributed *tsfresh*.

To distribute the calculation of features, we use a certain object, the Distributor class (located in the
:mod:`tsfresh.utilities.distribution` module).

Essentially, a Distributor organizes the application of feature calculators to data chunks.
It maps the feature calculators to the data chunks and then reduces them, meaning that it combines the results of the
individual mappings into one object, the feature matrix.

So, Distributor will, in the following order,

    1. calculate an optimal :python:`chunk_size`, based on the characteristics of the time series data
       (by :func:`~tsfresh.utilities.distribution.DistributorBaseClass.calculate_best_chunk_size`)

    2. split the time series data into chunks
       (by :func:`~tsfresh.utilities.distribution.DistributorBaseClass.partition`)

    3. distribute the application of the feature calculators to the data chunks
       (by :func:`~tsfresh.utilities.distribution.DistributorBaseClass.distribute`)

    4. combine the results into the feature matrix
       (by :func:`~tsfresh.utilities.distribution.DistributorBaseClass.map_reduce`)

    5. close all connections, shutdown all resources and clean everything
       (by :func:`~tsfresh.utilities.distribution.DistributorBaseClass.close`)

So, how can you use the Distributor to extract features with *tsfresh*?
You will have to pass :python:`distributor` as an argument to the :func:`~tsfresh.feature_extraction.extract_features`
method.


The following example shows how to define the MultiprocessingDistributor, which will distribute the calculations to a
local pool of threads:

.. code:: python

    from tsfresh.examples.robot_execution_failures import \
        download_robot_execution_failures, \
        load_robot_execution_failures
    from tsfresh.feature_extraction import extract_features
    from tsfresh.utilities.distribution import MultiprocessingDistributor

    # download and load some time series data
    download_robot_execution_failures()
    df, y = load_robot_execution_failures()

    # We construct a Distributor that will spawn the calculations
    # over four threads on the local machine
    Distributor = MultiprocessingDistributor(n_workers=4,
                                             disable_progressbar=False,
                                             progressbar_title="Feature Extraction")

    # just to pass the Distributor object to
    # the feature extraction, along with the other parameters
    X = extract_features(timeseries_container=df,
                         column_id='id',
                         column_sort='time',
                         distributor=Distributor)

The following example corresponds to the existing multiprocessing *tsfresh* API, where you just specify the number of
jobs, without the need to construct the Distributor:

.. code:: python

    from tsfresh.examples.robot_execution_failures import \
        download_robot_execution_failures, \
        load_robot_execution_failures
    from tsfresh.feature_extraction import extract_features

    download_robot_execution_failures()
    df, y = load_robot_execution_failures()

    X = extract_features(timeseries_container=df,
                         column_id='id',
                         column_sort='time',
                         n_jobs=4)

Using dask to distribute the calculations
'''''''''''''''''''''''''''''''''''''''''

We provide a Distributor for the `dask framework <https://dask.pydata.org/en/latest/>`_, where
*"Dask is a flexible parallel computing library for analytic computing."*

.. NOTE::
    This part of the documentation only handles parallelizing the computation using
    a dask cluster. The input and output are still pandas objects.
    If you want to use dask's capabilities to scale to data beyond your local
    memory, have a look at :ref:`large-data-label`.

Dask is a great framework to distribute analytic calculations into clusters.
It scales up and down, meaning that you can also use it on a singular machine.
The only thing that you will need to run *tsfresh* on a Dask cluster is the ip address and port number of the
`dask-scheduler <http://distributed.readthedocs.io/en/latest/setup.html>`_.

Let's say that your dask scheduler is running at ``192.168.0.1:8786``, then we can construct a
:class:`~sfresh.utilities.distribution.ClusterDaskDistributor` that connects to the scheduler and distributes the
time series data and the calculation to a cluster:

.. code:: python

    from tsfresh.examples.robot_execution_failures import \
        download_robot_execution_failures, \
        load_robot_execution_failures
    from tsfresh.feature_extraction import extract_features
    from tsfresh.utilities.distribution import ClusterDaskDistributor

    download_robot_execution_failures()
    df, y = load_robot_execution_failures()

    Distributor = ClusterDaskDistributor(address="192.168.0.1:8786")

    X = extract_features(timeseries_container=df,
                         column_id='id',
                         column_sort='time',
                         distributor=Distributor)

Compared to the :class:`~tsfresh.utilities.distribution.MultiprocessingDistributor` example from above, we only had to
change one line to switch from one machine to a whole cluster.
It is as easy as that.
By changing the Distributor you can easily deploy your application to run to a cluster instead of your workstation.

You can also use a local DaskCluster on your local machine to emulate a Dask network.
The following example shows how to setup a :class:`~tsfresh.utilities.distribution.LocalDaskDistributor` on a local cluster
of 3 workers:

.. code:: python

    from tsfresh.examples.robot_execution_failures import \
        download_robot_execution_failures, \
        load_robot_execution_failures
    from tsfresh.feature_extraction import extract_features
    from tsfresh.utilities.distribution import LocalDaskDistributor

    download_robot_execution_failures()
    df, y = load_robot_execution_failures()

    Distributor = LocalDaskDistributor(n_workers=3)

    X = extract_features(timeseries_container=df,
                         column_id='id',
                         column_sort='time',
                         distributor=Distributor)

Writing your own distributor
''''''''''''''''''''''''''''

If you want to use other framework instead of Dask, you will have to write your own Distributor.
To construct your custom Distributor, you need to define an object that inherits from the abstract base class
:class:`tsfresh.utilities.distribution.DistributorBaseClass`.
The :mod:`tsfresh.utilities.distribution` module contains more information about what you need to implement.
