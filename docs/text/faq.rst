FAQ
===


    1. **Does tsfresh support different time series lengths?**
       Yes, it supports different time series lengths. However, some feature calculators can demand a minimal length
       of the time series. If a shorter time series is passed to the calculator, normally a NaN is returned.


    2. **Is it possible to extract features from rolling/shifted time series?**
       Yes, the :func:`tsfresh.dataframe_functions.roll_time_series` function allows to conviniently create a rolled
       time series datframe from your data. You just have to transform your data into one of the supported tsfresh
       :ref:`data-formats-label`.
       Then, the :func:`tsfresh.dataframe_functions.roll_time_series` give you a DataFrame with the rolled time series,
       that you can pass to tsfresh.
       On the following page you can find a detailed description: :ref:`rolling-label`.
