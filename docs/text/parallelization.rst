.. _parallelization-label:

Parallelization
===============

The feature extraction as well as the feature selection offer the possibility of parallelization.
Out of the box both tasks are parallelized by tsfresh. However, the overhead introduced with the
parallelization should not be underestimated. Here we discuss the different settings to control
the parallelization. To achieve best results for your use-case you should experiment with the parameters.

Please let us know about your results tuning the below mentioned parameters! It will help improve this document as
well as the default settings.

Parallelization of Feature Selection
------------------------------------

We use a :class:`multiprocessing.Pool` to parallelize the calculation of the p-values for each feature. On
instantiation we set the Pool's number of worker processes to
`n_processes`. This field defaults to
the number of processors on the current system. We recommend setting it to the maximum number of available (and
otherwise idle) processors.

The chunksize of the Pool's map function is another important parameter to consider. It can be set via the
`chunksize` field. By default it is up to
:class:`multiprocessing.Pool` to decide on the chunksize.

Parallelization of Feature Extraction
-------------------------------------

For the feature extraction tsfresh exposes the parameters
`n_processes` and
`chunksize`. Both behave anlogue to the parameters
for the feature selection.

Additionally there are two options for how the parallelization is done:

1.  ``'per_kind'`` parallelizes the feature calculation per kind of time series.
2.  ``'per_sample'`` parallelizes per kind and per sample.

To enforce an option, either pass ``'per_kind'`` or ``'per_sample'`` as the ``parallelization=`` parameter of the
:func:`tsfresh.extract_features` function. By default the option is chosen with a rule of thumb:

If the number of different time series (kinds) is less than half of the number of available worker
processes (``n_processes``) then ``'per_sample'`` is chosen, otherwise ``'per_kind'``.

Generally, there is no perfect setting for all cases. On the one hand more parallelization can speed up the calculation
as the work is better distributed among the computers resources. On the other hand parallelization
introduces overheads such as copying data to the worker processes, splitting the data to enable the distribution and
combining the results.

Implementing the parallelization we observed the following aspects:

-   For small data sets the difference between parallelization per kind or per sample should be negligible.
-   For data sets with one kind of time series parallelization per sample results in a decent speed up that grows
    with the number of samples.
-   The more kinds of time series the data set contains, the more samples are necessary to make parallelization
    per sample worthwhile.
-   If the data set contains more kinds of time series than available cpu cores, parallelization per kind is
    the way to go.
