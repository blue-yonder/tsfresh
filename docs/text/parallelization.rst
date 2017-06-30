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
`n_jobs`. This field defaults to
the number of processors on the current system. We recommend setting it to the maximum number of available (and
otherwise idle) processors.

The chunksize of the Pool's map function is another important parameter to consider. It can be set via the
`chunksize` field. By default it is up to
:class:`multiprocessing.Pool` to decide on the chunksize.

Parallelization of Feature Extraction
-------------------------------------

For the feature extraction tsfresh exposes the parameters
`n_jobs` and `chunksize`. Both behave analogue to the parameters
for the feature selection.

To do performance studies and profiling, it sometimes quite useful to turn off parallelization at all. This can be
setting the parameter `n_jobs` to 0.
