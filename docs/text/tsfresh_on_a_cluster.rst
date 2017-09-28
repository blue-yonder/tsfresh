How to use tsfresh on a cluster
===============================

Sometimes the amount of time series data is too much to be processed on a singular machine.
Instead, it may be necessary to distribute the extraction of the features on a cluster.
Indeed, it is possible to extract features with *tsfresh* on a network of computational units.

This chapter will explain how you can distribute the feature extraction on a cluster.
Also, we will explain how you can construct custom Distributors, to user other distributed python frameworks.

The distributor class
'''''''''''''''''''''

To distribute the calculation of features, we use a certain object, the distributor.
Essentially, a Distributor organizes the application of feature calculators to data chunks.
It maps the feature calculators to the data chunks and then reduce them, meaning that it combines the results of the
individual mapping into one object, the feature matrix.

So, Distributor will, in the following order,

    1. calculate an optimal chunk_size based on the time series data
    2. split the time series data into chunks
    3. distribute the applying of the feature calculators on the data chunks, called map
    4. collect the results
    5. close all connections in the end
    6. combine the results, called reduce, into the feature matrix

So, how can you use such a Distributor? You will have to pass it into as the distributor argument to the
:func:`tsfresh.feature_extraction.extract_features` method.

.. code:: python

    from tsfresh.feature_extraction import extract_features
    from tsfresh.utilities.distribution import MultiprocessingDistributor

    # here we construct a Distributor that will spawn the calculations over four threads on the local machine
    Distributor = MultiprocessingDistributor(n_workers==4, disable_progressbar=False,
                                             progressbar_title="Feature Extraction")
    # we will just have to pass the Distributor object to the feature extraction, along the other parameters
    X = extract_features(distributor=Distributor, ...)


Using dask to distribute the calculations
'''''''''''''''''''''''''''''''''''''''''

We provide a


Writing your own distributor
''''''''''''''''''''''''''''

At the moment, we provide a Dask distributor, meaning that if you have a Dask cluster at hand, you can deploy *tsfresh*
on it.

To construct your custom Distributor, you will have to define an object that inherits from the abstract base class
:class:`tsfresh.utilities.distribution.DistributorBaseClass`.


