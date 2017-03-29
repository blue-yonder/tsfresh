.. _rolling-label:

How to handle rolling time series
=================================

Lets assume that we have a DataFrame of one of the tsfresh :ref:`data-formats-label`.
The "sort" column of such a container gives a sequential state to the individual measurements.
In the case of time series this can be the *time* dimension while in the case of spectra the order is given by the
*wavelength* or *frequency* dimensions.
We can exploit this sequence to generate more input data out of single time series, by *rolling* over the data.

Imagine the following situation:
You have the data of certain sensors (e.g. EEG measurements) as the base to classify patients into a healthy and not
healthy group (we oversimplify the problem here).
Lets say you have sensor data of 100 time steps, so you may extract features for the forecasting of the patients
healthiness by a classification algorithm.
If you also have measurements of the healthiness for those 100 time steps (this is the target vector), then you could
predict the healthiness of the patient in every time step, which essentially states a time series forecasting problem.
So, to do that, you want to extract features in every time step of the original time series while for example looking at
the last 10 steps.
A rolling mechanism creates such time series for every time step by creating sub time series of the sensor data of the
last 10 time steps.

Another example can be found in streaming data, e.g. in Industry 4.0 applications.
Here you typically get one new data row at a time and use this to for example predict machine failures. To train your model,
you could act as if you would stream the data, by feeding your classifier the data after one time step,
the data after the first two time steps etc.

Both examples imply, that you extract the features not only on the full data set, but also
on all temporal coherent subsets of data, which is the process of *rolling*. In tsfresh, this is implemented in the
function :func:`tsfresh.utilities.dataframe_functions.roll_time_series`.

The rolling mechanism takes a time series :math:`x` with its data rows :math:`[x_1, x_2, x_3, ..., x_n]`
and creates :math:`n` new time series :math:`\hat x^k`, each of them with a different consecutive part
of :math:`x`:

.. math::
    \hat x^k = [x_k, x_{k-1}, x_{k-2}, ..., x_1]

To see what this does in real-world applications, we look into the following example flat DataFrame in tsfresh format

+----+------+----+----+
| id | time | x  | y  |
+====+======+====+====+
| 1  | t1   | 1  | 5  |
+----+------+----+----+
| 1  | t2   | 2	 | 6  |
+----+------+----+----+
| 1  | t3   | 3	 | 7  |
+----+------+----+----+
| 1  | t4   | 4	 | 8  |
+----+------+----+----+
| 2  | t8   | 10 | 12 |
+----+------+----+----+
| 2  | t9   | 11 | 13 |
+----+------+----+----+

where you have measured the values from two sensors x and y for two different entities (id 1 and 2) in 4 or 2 time
steps (t1 to t9).

Now, we can use :func:`tsfresh.utilities.dataframe_functions.roll_time_series` to get consecutive sub-time series.
E.g. if you set `rolling` to 0, the feature extraction works on the original time series without any rolling.

So it extracts 2 set of features,

+----+------+----+----+
| id | time | x  | y  |
+====+======+====+====+
| 1  | t1   | 1  | 5  |
+----+------+----+----+
| 1  | t2   | 2	 | 6  |
+----+------+----+----+
| 1  | t3   | 3	 | 7  |
+----+------+----+----+
| 1  | t4   | 4	 | 8  |
+----+------+----+----+

and

+----+------+----+----+
| id | time | x  | y  |
+====+======+====+====+
| 2  | t8   | 10 | 12 |
+----+------+----+----+
| 2  | t9   | 11 | 13 |
+----+------+----+----+

If you set rolling to 1, the feature extraction works with all of the following time series:

+----+------+----+----+
| id | time | x  | y  |
+====+======+====+====+
| 1  | t1   | 1  | 5  |
+----+------+----+----+

+----+------+----+----+
| id | time | x  | y  |
+====+======+====+====+
| 1  | t1   | 1  | 5  |
+----+------+----+----+
| 1  | t2   | 2  | 6  |
+----+------+----+----+

+----+------+----+----+
| id | time | x  | y  |
+====+======+====+====+
| 1  | t1   | 1  | 5  |
+----+------+----+----+
| 1  | t2   | 2  | 6  |
+----+------+----+----+
| 1  | t3   | 3  | 7  |
+----+------+----+----+
| 2  | t8   | 10 | 12 |
+----+------+----+----+

+----+------+----+----+
| id | time | x  | y  |
+====+======+====+====+
| 1  | t1   | 1  | 5  |
+----+------+----+----+
| 1  | t2   | 2  | 6  |
+----+------+----+----+
| 1  | t3   | 3  | 7  |
+----+------+----+----+
| 1  | t4   | 4  | 8  |
+----+------+----+----+
| 2  | t8   | 10 | 12 |
+----+------+----+----+
| 2  | t9   | 11 | 13 |
+----+------+----+----+

If you set rolling to -1, you end up with features for the time series, rolled in the other direction

+----+------+----+----+
| id | time | x  | y  |
+====+======+====+====+
| 1  | t4   | 4  | 8  |
+----+------+----+----+

+----+------+----+----+
| id | time | x  | y  |
+====+======+====+====+
| 1  | t3   | 3  | 7  |
+----+------+----+----+
| 1  | t4   | 4  | 8  |
+----+------+----+----+

+----+------+----+----+
| id | time | x  | y  |
+====+======+====+====+
| 1  | t2   | 2  | 6  |
+----+------+----+----+
| 1  | t3   | 3  | 7  |
+----+------+----+----+
| 1  | t4   | 4  | 8  |
+----+------+----+----+
| 2  | t9   | 11 | 13 |
+----+------+----+----+

+----+------+----+----+
| id | time | x  | y  |
+====+======+====+====+
| 1  | t1   | 1  | 5  |
+----+------+----+----+
| 1  | t2   | 2  | 6  |
+----+------+----+----+
| 1  | t3   | 3  | 7  |
+----+------+----+----+
| 1  | t4   | 4  | 8  |
+----+------+----+----+
| 2  | t8   | 10 | 12 |
+----+------+----+----+
| 2  | t9   | 11 | 13 |
+----+------+----+----+

We only gave an example for the flat DataFrame format, but rolling actually works on all 3 :ref:`data-formats-label`
that are supported by tsfresh.