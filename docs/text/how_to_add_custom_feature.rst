How to add a custom feature
===========================

If you want to extract custom made features from your time series, tsfresh allows you to do so in a few
simple steps:

Step 1. Decide which type of feature you want to implement
----------------------------------------------------------

tsfresh supports two types of feature calculation methods:

    *1.* simple

    *2.* combiner

The difference lays in the number of features calculated for a singular time series.
The feature_calculator is simple if it returns one (*1.*) feature, and it is a combiner and returns multiple features (*2.*).
So if you want to add a singular feature, you should select *1.*, the simple feature calculator class.
If it is however, better to calculate multiple features at the same time (e.g., to perform auxiliary calculations only
once for all features), then you should choose type *2.*.


Step 2. Write the feature calculator
------------------------------------

Depending on which type of feature calculator you are implementing, you can use the following feature calculator skeletons:

1. simple features
~~~~~~~~~~~~~~~~~~

You can write a simple feature calculator that returns exactly one feature, without parameters as follows:

.. code:: python

    from tsfresh.feature_extraction.feature_calculators import set_property


    @set_property("fctype", "simple")
    def your_feature_calculator(x):
        """
        The description of your feature

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: bool, int or float
        """
        # Calculation of feature as float, int or bool
        result = f(x)
        return result

or with parameters:

.. code:: python

    @set_property("fctype", "simple"")
    def your_feature_calculator(x, p1, p2, ...):
        """
        Description of your feature

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param p1: description of your parameter p1
        :type p1: type of your parameter p1
        :param p2: description of your parameter p2
        :type p2: type of your parameter p2
        ...
        :return: the value of this feature
        :return type: bool, int or float
        """
        # Calculation of feature as float, int or bool
        f = f(x)
        return f


2. combiner features
~~~~~~~~~~~~~~~~~~~~

Alternatively, you can write a combiner feature calculator that returns multiple features as follows:

.. code:: python

    from tsfresh.utilities.string_manipulation import convert_to_output_format


    @set_property("fctype", "combiner")
    def your_feature_calculator(x, param):
        """
        Short description of your feature (should be a one liner as we parse the first line of the description)

        Long detailed description, add somme equations, add some references, what kind of statistics is the feature
        capturing? When should you use it? When not?

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param c: the time series name
        :type c: str
        :param param: contains dictionaries {"p1": x, "p2": y, ...} with p1 float, p2 int ...
        :type param: list
        :return: list of tuples (s, f) where s are the parameters, serialized as a string,
                 and f the respective feature value as bool, int or float
        :return type: pandas.Series
        """
        # Do some pre-processing if needed for all parameters
        # f is a function that calculates the feature value for each single parameter combination
        return [(convert_to_output_format(config), f(x, config)) for config in param]


Writing your own time-based feature calculators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Writing your own time-based feature calculators is no different than usual. Only two new properties must be set using the `@set_property` decorator:

* Adding ``@set_property("input", "pd.Series")`` tells the function that the input of the function is a ``pd.Series`` rather than a ``numpy`` array.
  This allows the index to be used automatically.
* Adding ``@set_property("index_type", pd.DatetimeIndex)`` tells the function that the input is a `DatetimeIndex`,
  allowing it to perform calculations based on time data types.

For example, if we want to write a function that calculates the time between the first and last measurement, it could look something like this:

.. code:: python

    @set_property("input", "pd.Series")
    @set_property("index_type", pd.DatetimeIndex)
    def timespan(x, param):
        ix = x.index

        # Get differences between the last timestamp and the first timestamp in seconds,
        # then convert to hours.
        times_seconds = (ix[-1] - ix[0]).total_seconds()
        return times_seconds / float(3600)


Step 3. Add custom settings for your feature
--------------------------------------------

Finally, you need to add your new custom feature to the extraction settings, otherwise it is not used
during extraction.
To do this, create a new settings object (by default, ``tsfresh`` uses the
:class:`tsfresh.feature_extraction.settings.ComprehensiveFCParameters`) and
add your function as a key to the dictionary.
As a value, either use ``None`` if your function does not need parameters or a list with the
parameters you want to use (as dictionaries).

.. code:: python

    settings = ComprehensiveFCParameters()
    settings[f] = [{"n": 1}, {"n": 2}]

After that, make sure you pass your newly created settings in the call to ``extract_features``.

Step 4. Make a pull request
---------------------------

We would be very happy if you contribute your custom features to tsfresh.

To do this, add your feature into the ``feature_calculators.py`` file and append your
feature (as a name) with safe default parameters to the ``name_to_param`` dictionary inside the
:class:`tsfresh.feature_extraction.settings.ComprehensiveFCParameters` constructor:

.. code:: python

    name_to_param.update({
        # here are the existing settings
        ...
        # Now the settings of your feature calculator
        "your_feature_calculator" = [{"p1": x, "p2": y, ...} for x,y in ...],
    })

Make sure, that the different feature extraction settings
(e.g. :class:`tsfresh.feature_extraction.settings.EfficientFCParameters`,
:class:`tsfresh.feature_extraction.settings.MinimalFCParameters` or
:class:`tsfresh.feature_extraction.settings.ComprehensiveFCParameters`) do include different sets of
feature calculators to use. You can control, which feature extraction settings object will include your new
feature calculator by giving your function attributes like "minimal" or "high_comp_cost". See the
classes in :mod:`tsfresh.feature_extraction.settings` for more information.

After that, add some tests and make a pull request to our `github repo <https://github.com/blue-yonder/tsfresh>`_.
We happily accept partly implemented feature calculators, which we can finalize together.
