.. _data-formats-label:

Data Formats
============

tsfresh offers three different options to specify the format of the time series data to use with the function
:func:`tsfresh.extract_features` (and all utility functions that expect a time series, for that
matter, like for example :func:`tsfresh.utilities.dataframe_functions.roll_time_series`).

Irrespective of the input format, tsfresh will always return the calculated features in the same output format
described below.

Typically, the input format options are :class:`pandas.DataFrame` objects, which we will discuss here, and also
Dask dataframes and PySpark computational graphs, which are discussed here :ref:`large-data-label`.

There are four important column types that
make up those DataFrames. Each will be described with an example from the robot failures dataset
(see :ref:`quick-start-label`).

:`column_id`: This column indicates which entities the time series belong to. Features will be extracted individually
    for each entity (id). The resulting feature matrix will contain one row per id.
    Each robot is a different entity, so each of it has a different id.

:`column_sort`: This column contains values which allow to sort the time series (e.g. time stamps).
    In general, it is not required to have equidistant time steps or the same time scale for the different ids and/or kinds.
    Some features might make however only sense for equidistant time stamps.
    If you omit this column, the DataFrame is assumed to be already sorted in ascending order.
    Each of the robot sensor measurements have a time stamp which is used as the `column_sort`.

Need only to be specified on some data formats (see below):

:`column_value`: This column contains the actual values of the time series.
    This corresponds to the measured values of different sensors on the robots.

:`column_kind`: This column indicates the names of the different time series types (e.g. different sensors in an
    industrial application as in the robot dataset).
    For each kind of time series the features are calculated individually.

Important: None of these columns is allowed to contain ``NaN``, ``Inf`` or ``-Inf`` values.

In the following paragrpahs, we describe the different input formats that are build based off those columns:

* A flat DataFrame
* A stacked DataFrame
* A dictionary of flat DataFrames

The difference between a flat and a stacked DataFrame is indicated by specifying (or not) the parameters
``column_value`` and ``column_kind`` in the :func:`tsfresh.extract_features` function.

If you are unsure which one to choose, try either the flat or stacked DataFrame.

Input Option 1. Flat DataFrame or Wide DataFrame
------------------------------------------------

If both ``column_value`` and ``column_kind`` are set to ``None``, the time series data is assumed to be in a flat
DataFrame. This means that each different time series must be saved as its own column.

Example: Imagine you record the values of time series x and y for different objects A and B for three different
times t1, t2 and t3. Your resulting DataFrame may look like this:

+----+------+----------+----------+
| id | time | x        | y        |
+====+======+==========+==========+
| A  | t1   | x(A, t1) | y(A, t1) |
+----+------+----------+----------+
| A  | t2   | x(A, t2) | y(A, t2) |
+----+------+----------+----------+
| A  | t3   | x(A, t3) | y(A, t3) |
+----+------+----------+----------+
| B  | t1   | x(B, t1) | y(B, t1) |
+----+------+----------+----------+
| B  | t2   | x(B, t2) | y(B, t2) |
+----+------+----------+----------+
| B  | t3   | x(B, t3) | y(B, t3) |
+----+------+----------+----------+

Now, you want to calculate some features with tsfresh so you would pass:

.. code:: python

    column_id="id", column_sort="time", column_kind=None, column_value=None

to the extraction function, to extract features separately for all ids and separately for the x and y values.
You can also omit the ``column_kind=None, column_value=None`` as this is the default.

Input Option 2. Stacked DataFrame or Long DataFrame
---------------------------------------------------

If both ``column_value`` and ``column_kind`` are set, the time series data is assumed to be a stacked DataFrame.
This means that there are no different columns for the different types of time series.
This representation has several advantages over the flat Data Frame.
For example, the time stamps of the different time series do not have to align.

It does not contain different columns for the different types of time series but only one
value column and a kind column. Following with our previous example, the dataframe would look like this:

+----+------+------+----------+
| id | time | kind | value    |
+====+======+======+==========+
| A  | t1   | x    | x(A, t1) |
+----+------+------+----------+
| A  | t2   | x    | x(A, t2) |
+----+------+------+----------+
| A  | t3   | x    | x(A, t3) |
+----+------+------+----------+
| A  | t1   | y    | y(A, t1) |
+----+------+------+----------+
| A  | t2   | y    | y(A, t2) |
+----+------+------+----------+
| A  | t3   | y    | y(A, t3) |
+----+------+------+----------+
| B  | t1   | x    | x(B, t1) |
+----+------+------+----------+
| B  | t2   | x    | x(B, t2) |
+----+------+------+----------+
| B  | t3   | x    | x(B, t3) |
+----+------+------+----------+
| B  | t1   | y    | y(B, t1) |
+----+------+------+----------+
| B  | t2   | y    | y(B, t2) |
+----+------+------+----------+
| B  | t3   | y    | y(B, t3) |
+----+------+------+----------+

Then you would set:

.. code:: python

    column_id="id", column_sort="time", column_kind="kind", column_value="value"

to end up with the same extracted features.
You can also omit the value column and let ``tsfresh`` deduce it automatically.


Input Option 3. Dictionary of flat DataFrames
---------------------------------------------

Instead of passing a DataFrame which must be split up by its different kinds by tsfresh, you can also give a
dictionary mapping from the kind as string to a DataFrame containing only the time series data of that kind.
So essentially you are using a singular DataFrame for each kind of time series.

The data from the example can be split into two DataFrames resulting in the following dictionary:

{ "x":

    +----+------+----------+
    | id | time | value    |
    +====+======+==========+
    | A  | t1   | x(A, t1) |
    +----+------+----------+
    | A  | t2   | x(A, t2) |
    +----+------+----------+
    | A  | t3   | x(A, t3) |
    +----+------+----------+
    | B  | t1   | x(B, t1) |
    +----+------+----------+
    | B  | t2   | x(B, t2) |
    +----+------+----------+
    | B  | t3   | x(B, t3) |
    +----+------+----------+

,
"y":

   +----+------+----------+
   | id | time | value    |
   +====+======+==========+
   | A  | t1   | y(A, t1) |
   +----+------+----------+
   | A  | t2   | y(A, t2) |
   +----+------+----------+
   | A  | t3   | y(A, t3) |
   +----+------+----------+
   | B  | t1   | y(B, t1) |
   +----+------+----------+
   | B  | t2   | y(B, t2) |
   +----+------+----------+
   | B  | t3   | y(B, t3) |
   +----+------+----------+

}

You would pass this dictionary to tsfresh together with the following arguments:

.. code:: python

    column_id="id", column_sort="time", column_kind=None, column_value="value":


In this case we do not need to specify the kind column as the kind is the respective dictionary key.

Output Format
-------------

The resulting feature matrix, containing the extracted features, is the same for all three input options.
It will always be a :class:`pandas.DataFrame` with the following layout:

+----+-------------+-----+-------------+-------------+-----+-------------+
| id | x_feature_1 | ... | x_feature_N | y_feature_1 | ... | y_feature_N |
+====+=============+=====+=============+=============+=====+=============+
| A  | ...         | ... | ...         | ...         | ... | ...         |
+----+-------------+-----+-------------+-------------+-----+-------------+
| B  | ...         | ... | ...         | ...         | ... | ...         |
+----+-------------+-----+-------------+-------------+-----+-------------+

where the x features are calculated using all x values (independently for A and B), the y features using all y values
(independently for A and B), and so on.

This DataFrame is also the expected input format to the feature selection algorithms used by tsfresh (e.g. the
:func:`tsfresh.select_features` function).
