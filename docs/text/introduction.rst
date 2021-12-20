Introduction
============

Why tsfresh?
------------

tsfresh is used for systematic feature engineering from time-series and other sequential data [1]_.
These data have in common that they are ordered by an independent variable.
The most common independent variable is time (time series).
Other examples for sequential data are reflectance and absorption spectra,
which have wavelength as their ordering dimension.
In order keeps things simple, we are simplify referring to all different types of sequential data as time-series.

.. image:: ../images/introduction_ts_exa.png
   :scale: 70 %
   :alt: the time series
   :align: center

(and yes, it is pretty cold!)

Now you want to calculate different characteristics such as the maximum or minimum temperature, the average temperature
or the number of temporary temperature peaks:

.. image:: ../images/introduction_ts_exa_features.png
   :scale: 70 %
   :alt: some characteristics of the time series
   :align: center

Without tsfresh, you would have to calculate all those characteristics manually; tsfresh automates this process
calculating and returning all those features automatically.

In addition, tsfresh is compatible with the Python libraries :mod:`pandas` and :mod:`scikit-learn`, so you can easily
integrate the feature extraction with your current routines.

What can we do with these features?
-----------------------------------

The extracted features can be used to describe the time series, i.e., often these features give new insights into the
time series and their dynamics. They can also be used to cluster time series and to train machine learning models that
perform classification or regression tasks on time series.

The tsfresh package has been successfully used in the following projects:

    * prediction of steel billets quality during a continuous casting process [2]_,
    * activity recognition from synchronized sensors [3]_,
    * volcanic eruption forecasting [4]_,
    * authorship attribution from written text samples [5]_,
    * characterisation of extrasolar planetary systems from time-series with missing data [6]_,
    * sensor anomaly detection [7]_,
    * and `many many more <https://scholar.google.de/scholar?cites=365611925060572663>`_.

What can't we do with tsfresh?
------------------------------

Currently, tsfresh is not suitable:

    * for streaming data (by streaming data we mean data that is usually used for online operations, while time series data is usually used for offline operations)
    * to train models on the extracted features (we do not want to reinvent the wheel, to train machine learning models check out the Python package
      `scikit-learn <http://scikit-learn.org/stable/>`_)
    * for usage with highly irregular time series; tsfresh uses timestamps only to order observations, while many features are interval-agnostic (e.g., number of peaks) and can be determined for any series, some otherfeatures (e.g., linear trend) assume equal spacing in time, and should be used with care when this assumption is not met.

However, some of these use cases could be implemented, if you have an application in mind, open
an issue at `<https://github.com/blue-yonder/tsfresh/issues>`_, or feel free to contact us.

What else is out there?
-----------------------

There is a matlab package called `hctsa <https://github.com/benfulcher/hctsa>`_ which can be used to automatically
extract features from time series.
It is also possible to use hctsa from within Python through the `pyopy <https://github.com/strawlab/pyopy>`_
package.
Other available packagers are `featuretools <https://www.featuretools.com/>`_, `FATS <http://isadoranun.github.io/tsfeat/>`_ and `cesium <http://cesium-ml.org/>`_.

References
----------

   .. [1] Christ, M., Braun, N., Neuffer, J. and Kempa-Liehr A.W. (2018).
          *Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package)*.
          Neurocomputing 307 (2018) 72-77,
          `doi: 10.1016/j.neucom.2018.03.067 <https://doi.org/10.1016/j.neucom.2018.03.067>`_.
   .. [2] Christ, M., Kempa-Liehr, A.W. and Feindt, M. (2016).
          *Distributed and parallel time series feature extraction for industrial big data applications*.
          Asian Conference on Machine Learning (ACML), Workshop on Learning on Big Data (WLBD).
          `<https://arxiv.org/abs/1610.07717v1>`_.
   .. [3] Kempa-Liehr, A.W., Oram, J., Wong, A., Finch, M. and Besier, T. (2020).
          *Feature engineering workflow for activity recognition from synchronized inertial measurement units*.
          In: Pattern Recognition. ACPR 2019. Ed. by M. Cree et al. Vol. 1180.
          Communications in Computer and Information Science (CCIS).
          Singapore: Springer 2020, 223–231.
          `doi: 10.1007/978-981-15-3651-9_20 <https://doi.org/10.1007/978-981-15-3651-9_20>`_.
   .. [4] D. E. Dempsey, S. J. Cronin, S. Mei, and A. W. Kempa-Liehr (2020).
          *Automatic precursor recognition and real-time forecasting of sudden explosive volcanic eruptions at Whakaari, New Zealand*.
          Nature Communications 11.3562, pp. 1–8.
          `doi: 10.1038/s41467-020-17375-2 <https://dx.doi.org/10.1038/s41467-020-17375-2>`_.
   .. [5] Tang, Y., Blincoe, K., Kempa-Liehr, A.W. (2020).
          *Enriching Feature Engineering for Short Text Samples by Language Time Series Analysis*.
          EPJ Data Science 9.26 (2020), 1–59.
          `doi: 10.1140/epjds/s13688-020-00244-9 <https://doi.org/10.1140/epjds/s13688-020-00244-9>`_.
   .. [6] Kennedy, A., Gemma, N., Rattenbury, N., Kempa-Liehr, A.W. (2021).
          *Modelling the projected separation of microlensing events using systematic time-series feature engineering*.
          Astronomy and Computing 35.100460 (2021), 1–14,
          `doi: 10.1016/j.ascom.2021.100460 <https://doi.org/10.1016/j.ascom.2021.100460>`_.
   .. [7] Hui Yie Teh, Kevin I-Kai Wang, and Andreas W. Kempa-Liehr (2021).
          *Expect the Unexpected: Unsupervised feature selection for automated sensor anomaly detection*.
          IEEE Sensors Journal 15.16, pp. 18033–18046.
          `doi: 10.1109/JSEN.2021.3084970 <https://doi.org/10.1109/JSEN.2021.3084970>`_.
