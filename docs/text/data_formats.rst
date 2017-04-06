.. _data-formats-label:

Data Formats
============

tsfresh offers three different options to specify the time series data to be used in the
:func:`tsfresh.extract_features` function (and all utility functions that expect a time series, e.g. the
:func:`tsfresh.utilities.dataframe_functions.roll_time_series` function).

Irrespective of the input format, tsfresh will always return the calculated features in the same output format
described below.

All three input format options consist of :class:`pandas.DataFrame` objects. There are four important column types that
make up those DataFrames. Each will be described with an example from the robot failures dataset
(see :ref:`quick-start-label`).

Mandatory:

:`column_id`: This column indicates which entities the time series belong to. Features will be extracted individually
    for each entity. The resulting feature matrix will contain one row per entity.
    Each robot is a different entity, so each of it has a different id.
:`column_value`: This column contains the actual values of the time series.
    This corresponds to the measured values for different the sensors on the robots.

Optional (but strongly recommended to specify if you have this column):

:`column_sort`: This column contains values which allow to sort the time series (e.g. time stamps). It is not required
    to have equidistant time steps or the same time scale for the different ids and/or kinds.
    If you omit this column, the DataFrame is assumed to be already sorted in increasing order.
    The robot sensor measurements each have a time stamp which is used in this column.

    Please note that none of the algorithms of tsfresh uses the actual values in this time column - but only their
    sorting order.

Optional:

:`column_kind`: This column indicates the names of the different time series types (E.g. different sensors in an
    industrial application as in the robot dataset).
    For each kind of time series the features are calculated individually.


Important: None of these columns is allowed to contain any ``NaN``, ``Inf`` or ``-Inf`` values.

In the following we describe the different input formats, that are build on those columns:
    * A flat DataFrame
    * A stacked DataFrame
    * A dictionary of flat DataFrames

The difference between a flat and a stacked DataFrame is indicated by specifying or not specifying the parameters
`column_value` and `column_kind` in the :func:`tsfresh.extract_features` function.

If you do not know which one to choose, you probably want to try out the flat or stacked DataFrame.

Input Option 1. Flat DataFrame
------------------------------

If both `column_value` and `column_kind` are set to ``None``, the time series data is assumed to be in a flat
DataFrame. This means that each different time series must be saved as its own column.

Example: Imagine you record the values of time series x and y for different objects A and B for three different
times t1, t2 and t3. Now you want to calculate some feature with tsfresh. Your resulting DataFrame may look
like this:

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

and you would pass

.. code:: python

    column_id="id", column_sort="time", column_kind=None, column_value=None

to the extraction functions, to extract features separately for all ids and separately for the x and y values.

Input Option 2. Stacked DataFrame
---------------------------------

If both `column_value` and `column_kind` are set, the time series data is assumed to be a stacked DataFrame.
This means that there are no different columns for the different types of time series.
This representation has several advantages over the flat Data Frame.
For example, the time stamps of the different time series do not have to align.

It does not contain different columns for the different types of time series but only one
value column and a kind column. The example from above would look like this:

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

Then you would set

.. code:: python

    column_id="id", column_sort="time", column_kind="kind", column_value="value"

to end up with the same extracted features as above.


Input Option 3. Dictionary of flat DataFrames
---------------------------------------------

Instead of passing a DataFrame which must be split up by its different kinds by tsfresh, you can also give a
dictionary mapping from the kind as string to a DataFrame containing only the time series data of that kind.
So essentially you are using a singular DataFrame for each kind of time series.

The data from the example can be split into two DataFrames resulting in the following dictionary

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

The resulting feature matrix for all three input options will be the same.
It will always be a :class:`pandas.DataFrame` with the following layout

+----+-------------+-----+-------------+-------------+-----+-------------+
| id | x_feature_1 | ... | x_feature_N | y_feature_1 | ... | y_feature_N |
+====+=============+=====+=============+=============+=====+=============+
| A  | ...         | ... | ...         | ...         | ... | ...         |
+----+-------------+-----+-------------+-------------+-----+-------------+
| B  | ...         | ... | ...         | ...         | ... | ...         |
+----+-------------+-----+-------------+-------------+-----+-------------+

where the x features are calculated using all x values (independently for A and B), y features using all y values
and so on.

This form of DataFrame is also the expected input format to the feature selection algorithms (e.g. the
:func:`tsfresh.select_features` function).