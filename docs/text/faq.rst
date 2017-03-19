FAQ
===


    1. *Does tsfresh support different time series lengths?*
       Yes, it supports different time series lengths. However, some feature calculators can demand a minimal length
       of the time series. If a shorter time series is passed to the calculator, normally a NaN is returned.


    2. *Is it possible to extract features from rolling/shifted time series?*
       Yes, there is the option `rolling` for the :func:`tsfresh.feature_extraction.extract_features` function.
       Set it to a non-zero value to enable rolling. In the moment, this just rolls the input data into
       as many time series as there are time steps - so there is no internal optimization for rolling calculations.
       Please see :ref:`rolling-label` for more information.
