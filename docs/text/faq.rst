FAQ
===


    1. **Does tsfresh support different time series lengths?**

       Yes, it supports different time series lengths. However, some feature calculators can demand a minimal length
       of the time series. If a shorter time series is passed to the calculator, a NaN is returned for those
       features.



    2. **Is it possible to extract features from rolling/shifted time series?**

       Yes, the :func:`tsfresh.dataframe_functions.roll_time_series` function allows to conveniently create a rolled
       time series dataframe from your data. You just have to transform your data into one of the supported tsfresh
       :ref:`data-formats-label`.
       Then, the :func:`tsfresh.dataframe_functions.roll_time_series` will return a DataFrame with the rolled time series,
       that you can pass to tsfresh.
       You can find more details here: :ref:`forecasting-label`.


    3. **How can I use tsfresh with windows?**

       We recommend to use `Anaconda <https://www.continuum.io/downloads#windows>`_. After installation, open the
       Anaconda Prompt, create an environment and set up tsfresh
       (Please be aware that we're using multiprocessing, which can be `problematic <http://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing>`_.):

       .. code:: Bash

           conda create -n ENV_NAME python=VERSION
           conda install -n ENV_NAME pip requests numpy pandas scipy statsmodels patsy scikit-learn tqdm
           activate ENV_NAME
           pip install tsfresh


    4. **Does tsfresh support different sampling rates in the time series?**

        Yes! The feature calculators in tsfresh do not care about the sampling frequency.
        You will have to use the second input format, the stacked DataFramed (see :ref:`data-formats-label`)


    5. **Does tsfresh support irregularly spaced time series?**

	Yes, but be careful. As its name suggests, the ``column_sort`` (i.e., timestamp) parameter is only used to sort observations.
	Beyond sorting, tsfresh does not use the timestamp in calculations.
	While many features do not need a timestamp (or only need it for ordering), others will assume that observations are evenly spaced in time (e.g., one second between each observation).
	Since tsfresh ignores spacing, care should be taken when selecting features to use with a highly irregular series.
