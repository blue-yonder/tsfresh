FAQ
===


    1. **Does tsfresh support different time series lengths?**

       Yes, it supports different time series lengths. However, some feature calculators can demand a minimal length
       of the time series. If a shorter time series is passed to the calculator, a NaN is returned for those
       features.



    2. **Is it possible to extract features from rolling/shifted time series?**

       Yes, the :func:`tsfresh.dataframe_functions.roll_time_series` function allows to conviniently create a rolled
       time series datframe from your data. You just have to transform your data into one of the supported tsfresh
       :ref:`data-formats-label`.
       Then, the :func:`tsfresh.dataframe_functions.roll_time_series` give you a DataFrame with the rolled time series,
       that you can pass to tsfresh.
       On the following page you can find a detailed description: :ref:`rolling-label`.


    3. **How can I tsfresh with windows?**

       We recommend to use `Anaconda <https://www.continuum.io/downloads#windows>`_. After installing open the
       Anaconda prompt, create an environment and set up tsfresh:
       ::
           conda create -n ENV_NAME python=VERSION
           conda install -n ENV_NAME pip requests numpy pandas scipy statsmodels patsy scikit-learn future six tqdm
           activate ENV_NAME
           pip install tsfresh

       Please be aware that we're using multiprocessing, which can be `problematic <http://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing>`_.
