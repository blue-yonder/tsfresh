.. _large-data-label:

Large Input Data
================

If you are working with large time series data, you are probably facing multiple problems.
The two most important ones are:

* long execution times for feature extraction
* large memory consumption, even beyond what a single machine can handle

To solve the first problem, you can parallelize the computation as described in :ref:`tsfresh-on-a-cluster-label`.
Note, that parallelization on your local computer is already turned on by default.

However, for larger data sets you need to handle both problems at the same time.
You have multiple options to do so, which we will discuss in the following paragraphs.

Dask - the simple way
---------------------

*tsfresh* accepts a `dask dataframe <https://docs.dask.org/en/latest/dataframe.html>`_ instead of a
pandas dataframe as input for the :func:`tsfresh.extract_features` function.
Dask dataframes allow you to scale your computation beyond your local memory (via partitioning the data internally)
and even to large clusters of machines.
Its dataframe API is very similar to pandas dataframes and might even be a drop-in replacement.

All arguments discussed in :ref:`data-formats-label` are also valid for dask dataframes.
The input data will be transformed into the correct format for *tsfresh* using dask methods
and the feature extraction will be added as additional computations to the computation graph.
You can then add additional computations to the result or trigger the computation as usual with ``.compute()``.

.. NOTE::

    The last step of the feature extraction is to bring all features into a tabular format.
    Especially for very large data samples, this computation can be a large
    performance bottleneck.
    We therefore recommend to turn the pivoting off, if you do not really need it
    and work with the un-pivoted data as much as possible.

For example, to read in data from parquet and do the feature extraction:

.. code::

    import dask.dataframe as dd
    from tsfresh import extract_features

    df = dd.read_parquet(...)

    X = extract_features(df,
                         column_id="id",
                         column_sort="time",
                         pivot=False)

    result = X.compute()

Dask - more control
-------------------

The feature extraction method needs to perform some data transformations before it
can call the actual feature calculators.
If you want to optimize your data flow, you might want to have more control on how
exactly the feature calculation is added to you dask computation graph.

Therefore, it is also possible to add the feature extraction directly:


.. code::

    from tsfresh.convenience.bindings import dask_feature_extraction_on_chunk
    features = dask_feature_extraction_on_chunk(df_grouped,
                                                column_id="id",
                                                column_kind="kind",
                                                column_sort="time",
                                                column_value="value")

In this case however, ``df_grouped`` must already be in the correct format.
Check out the documentation of :func:`tsfresh.convenience.bindings.dask_feature_extraction_on_chunk`
for more information.
No pivoting will be performed in this case.

PySpark
-------

Similar to dask, it is also possible to pass the feature extraction into a Spark
computation graph.
You can find more information in the documentation of :func:`tsfresh.convenience.bindings.spark_feature_extraction_on_chunk`.
