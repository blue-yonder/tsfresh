How to add a custom feature
===========================

It may be beneficial to add a custom feature to those that are calculated by tsfresh. To do so, one has to follow four
simple steps:

Step 1. Decide which type of feature you want to implement
----------------------------------------------------------

In tsfresh we differentiate between two types of feature calculation methods

    *1.* simple

    *2.* combiner

The difference lays in the number of calculated features for a singular time series.
The feature_calculator returns either one (*1.*) or multiple features (*2.*).
So if you want to add a singular feature stick with *1.*, the simple feature calculator class.
If it is beneficial to calculate multiples features at the same time (to e.g. perform auxiliary calculations only once
for all features), stick with type *2.*.


Step 2. Write the feature calculator
------------------------------------

Depending on which type of feature you are implementing, you can use the following feature calculator skeletons:

*1.* simple features

You can write such a simple feature calculator, that returns exactly one feature, without parameter

.. code:: python

    from tsfresh.feature_extraction.feature_calculators import set_propert


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
        f = f(x)
        return f

or with parameter

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


*2.* combiner features

.. code:: python

    @set_property("fctype", "combiner")
    def your_feature_calculator(x, param):
        """
        Description of your feature

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param c: the time series name
        :type c: str
        :param param: contains dictionaries {"p1": x, "p2": y, ...} with p1 float, p2 int ...
        :type param: list
        :return: list of tuples (s, f) where s are the parameters, serialized as a string, and f the respective feature
            value as bool, int or float
        :return type: pandas.Series
        """
        # s is a function that serializes the config
        # f is a function that calculates the feature value for the config
        return [(s(config), f(x, config)) for config in param]


After implementing the feature calculator, please add it to the :mod:`tsfresh.feature_extraction.feature_calculators`
submodule. tsfresh will only find feature calculators that are in this submodule.


Step 3. Add custom settings for your feature
--------------------------------------------

Finally, you have to add custom settings if your feature is a simple or combiner feature with parameters. To do so,
just append your feature with sane default parameters to the ``name_to_param`` dictionary inside the
:class:`tsfresh.feature_extraction.settings.ComprehensiveFCParameters` constructor:

.. code:: python

    name_to_param.update({
        # here are the existing settings
        ...
        # Now the settings of your feature calculator
        "your_feature_calculator" = [{"p1": x, "p2": y, ...} for x,y in ...],
    })


That is it, tsfresh will calculate your feature the next time you run it.

Please make sure, that the different feature extraction settings
(e.g. :class:`tsfresh.feature_extraction.settings.EfficientFCParameters`,
:class:`tsfresh.feature_extraction.settings.MinimalFCParameters` or
:class:`tsfresh.feature_extraction.settings.ComprehensiveFCParameters`) do include different sets of
feature calculators to use. You can control, which feature extraction settings object will include your new
feature calculator by giving your function attributes like "minimal" or "high_comp_cost". Please see the
classes in :mod:`tsfresh.feature_extraction.settings` for more information.


Step 4. Add a pull request
--------------------------

We would very happy if you contribute your implemented features to tsfresh. So make sure to create a pull request at our
`github page <https://github.com/blue-yonder/tsfresh>`_. We happily accept partly implemented feature calculators, which
we can finalize collaboratively.
