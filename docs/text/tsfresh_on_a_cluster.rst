.. role:: python(code)
    :language: python

How to use tsfresh on a cluster
===============================

Sometimes the volume of your time series data is too large to be processed on a singular machine.
Instead, it may be necessary to distribute the extraction of the features to a cluster.
Indeed, it is possible to extract features with *tsfresh* in a distributed fashion.

This page will explain how to setup a distributed *tsfresh*.

The distributor class
'''''''''''''''''''''

To distribute the calculation of features, we use a certain object, the Distributor class (contained in
:mod:`tsfresh.utilities.distribution` module).
Essentially, a Distributor organizes the application of feature calculators to data chunks.

It maps the feature calculators to the data chunks and then reduces them, meaning that it combines the results of the
individual mapping into one object, the feature matrix.

So, Distributor will, in the following order,

    1. calculate an optimal `chunk_size` based on the time series data (by :func:~`tsfresh.utilities.distribution.DistributorBaseClass.\_calculate_best_chunk_size`)
    2. split the time series data into chunks
    3. distribute the applying of the feature calculators on the data chunks, called map
    4. combine the results, called reduce, into the feature matrix
    5. close all connections, clean the network (by :func:`tsfresh.utilities.distribution.DistributorBaseClass.close`)

So, how can you use such a Distributor to extract features?
You will have to pass it into as the :python:`distributor` argument to the :func:~`tsfresh.feature_extraction.extract_features`
method.~


The following example shows how to define the MultiprocessingDistributor, which will distribute the calculations to a
local pool of threads:~

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

    # we will just have to pass the Distributor object to
    # the feature extraction, along the other parameters
    X = extract_features(timeseries_container=df,
                         column_id='id', column_sort='time',
                         distributor=Distributor)

This example actually corresponds to the existing multiprocessing *tsfresh* version, where you just specify the number of
jobs, without the need to construct the Distributor:

.. code:: python

    from tsfresh.examples.robot_execution_failures import \
        download_robot_execution_failures, \
        load_robot_execution_failures
    from tsfresh.feature_extraction import extract_features
    from tsfresh.utilities.distribution import MultiprocessingDistributor

    # download and load some time series data
    download_robot_execution_failures()
    df, y = load_robot_execution_failures()

    # we will just have to pass the Distributor object to
    # the feature extraction, along the other parameters
    X = extract_features(timeseries_container=df,
                         column_id='id', column_sort='time',
                         n_jobs=4)

Using dask to distribute the calculations
'''''''''''''''''''''''''''''''''''''''''

We provide distributor for the `dask framework <https://dask.pydata.org/en/latest/>`_, where
*"Dask is a flexible parallel computing library for analytic computing."*

Dask is a great framework to distribute analytic calculations to a cluster.
You can also use it on a singular machine.
The only thing that you will need to run *tsfresh* on a Dask cluster is the ip address and port number of the
`dask-scheduler <http://distributed.readthedocs.io/en/latest/setup.html>`_.

Lets say that your dask scheduler is running at ``192.168.0.1:8786``, then in the

.. code:: python

    from tsfresh.examples.robot_execution_failures import \
        download_robot_execution_failures, \
        load_robot_execution_failures
    from tsfresh.feature_extraction import extract_features
    from tsfresh.utilities.distribution import ClusterDaskDistributor

    download_robot_execution_failures()
    df, y = load_robot_execution_failures()

    # We construct a Distributor that will distribute the calculations on a
    # Dask Cluster
    Distributor = MultiprocessingDistributor(address="192.168.0.1:8786")

    X = extract_features(timeseries_container=df,
                         column_id='id', column_sort='time',
                         distributor=Distributor)

Compare to the :mod:`tsfresh.utilities.distribution.MultiprocessingDistributor` example from above, we only had to
change one line.
It is as easy as that.
By changing the Distributor you can easily deploy your application to run on a cluster instead of your workstation.


Writing your own distributor
''''''''''''''''''''''''''''

To construct your custom Distributor, you will have to define an object that inherits from the abstract base class
:class:`tsfresh.utilities.distribution.DistributorBaseClass`.
The :mod:`tsfresh.utilities.distribution` contains more information about what you will need to implement.


