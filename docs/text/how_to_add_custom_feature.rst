How to add a custom feature
===========================

It may be beneficial to add a custom feature to those that are calculated by tsfresh. To do so, one has to adapt certain
steps:

Step 1. Decide which type of feature you want to implement
----------------------------------------------------------

In tsfresh we differentiate between three types of feature calculation methods

    *1.* aggregate features without parameter

    *2.* aggregate features with parameter

    *3.* apply features with parameters

So if you want to add a singular feature with out any parameters, stick with *1.*, the aggregate feature without
parameters.

Then, if your features can be calculated independently for each parameter, stick with type *2.*, the
aggregate features with parameters.


If both cases from above do not apply, so it is beneficial to calculate the features for the different
parameter settings at the same time (to e.g. perform auxiliary calculations only once for all features), stick with
type *3.*, the apply features with parameters.


Step 2. Write the feature calculator
------------------------------------

Depending on which type of feature you are implementing, you can use the following feature calculator skeletons:

*1.* aggregate features without parameter

.. code:: python

    @set_property("fctype", "aggregate")
    def your_feature_calculator(x):
        """
        The description of your feature

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: bool or float
        """
        # Calculation of feature as float, int or bool
        f = f(x)
        return f



*2.* aggregate features with parameter

.. code:: python

    @set_property("fctype", "aggregate_with_parameters")
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
        :return type: bool or float
        """
        # Calculation of feature as float, int or bool
        f = f(x)
        return f


*3.* apply features with parameters

.. code:: python

    @set_property("fctype", "apply")
    def your_feature_calculator(x, c, param):
        """
        Description of your feature

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param c: the time series name
        :type c: str
        :param param: contains dictionaries {"p1": x, "p2": y, ...} with p1 float, p2 int ...
        :type param: list
        :return: the different feature values
        :return type: pandas.Series
        """
        # Calculation of feature as pandas.Series s, the index is the name of the feature
        s = f(x)
        return s


After implementing the feature calculator, please add it to the :mod:`tsfresh.feature_extraction.extraction` submodule.
tsfresh will only find feature calculators that are in this submodule.


Step 3. Add custom settings for your feature
--------------------------------------------

Finally, you have to add custom settings if your feature is a apply or aggregate feature with parameters. To do so,
just append your parameters to the ``name_to_param`` dictionary inside the
:func:`tsfresh.feature_extraction.settings.set_default_parameters` method:

.. code:: python

    name_to_param.update({
        # here are the existing settings
        ...
        # Now the settings of your feature calculator
        "your_feature_calculator" = [{"p1": x, "p2": y, ...} for x,y in ...],
    })


That is it, tsfresh will calculate your feature the next time you run it.

