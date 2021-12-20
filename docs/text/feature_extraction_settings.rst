Feature extraction settings
===========================

When starting a new data science project involving time series you probably want to start by extracting a
comprehensive set of features. Later you can identify which features are relevant for the task at hand.
In the final stages, you probably want to fine tune the parameter of the features to fine tune your models.

You can do all those things with tsfresh. So, you need to know how to control which features are calculated by tsfresh
and how one can adjust the parameters. In this section, we will clarify this.

For the lazy: Just let me calculate some features!
--------------------------------------------------

To calculate a comprehensive set of features, call the :func:`tsfresh.extract_features` method without
passing a ``default_fc_parameters`` or ``kind_to_fc_parameters`` object. This way you will be using the default options,
which will use all the feature calculators in this package, that we consider are OK to return by default.

For the advanced: How do I set the parameters for all kind of time series?
----------------------------------------------------------------------------

After digging deeper into your data, you maybe want to calculate more of a certain type of feature and less of another
type. So, you need to use custom settings for the feature extractors. To do that with tsfresh you will have to use a
custom settings object:

>>> from tsfresh.feature_extraction import ComprehensiveFCParameters
>>> settings = ComprehensiveFCParameters()
>>> # Set here the options of the settings object as shown in the paragraphs below
>>> # ...
>>> from tsfresh.feature_extraction import extract_features
>>> extract_features(df, default_fc_parameters=settings)

The ``default_fc_parameters`` is expected to be a dictionary which maps feature calculator names
(the function names you can find in the :mod:`tsfresh.feature_extraction.feature_calculators` file) to a list
of dictionaries, which are the parameters with which the function will be called (as key value pairs). Each
function-parameter combination that is in this dict will be called during the extraction and will produce a feature.
If the function does not take any parameters, the value should be set to `None`.

For example:

.. code:: python

    fc_parameters = {
        "length": None,
        "large_standard_deviation": [{"r": 0.05}, {"r": 0.1}]
    }

will produce three features: one by calling the
:func:`tsfresh.feature_extraction.feature_calculators.length` function without any parameters and two by calling
:func:`tsfresh.feature_extraction.feature_calculators.large_standard_deviation` with `r = 0.05` and `r = 0.1`.

So you can control which features will be extracted, by adding or removing either keys or parameters from this dict.
It is as easy as that.
If you decide not to calculate the length feature here, you delete it from the dictionary:

.. code:: python

    del fc_parameters["length"]

And now, only the two other features are calculated.

For convenience, three dictionaries are predefined and can be used right away:

* :class:`tsfresh.feature_extraction.settings.ComprehensiveFCParameters`: includes all features without parameters and
  all features with parameters, each with different parameter combinations. This is the default for `extract_features`
  if you do not hand in a `default_fc_parameters` at all.
* :class:`tsfresh.feature_extraction.settings.MinimalFCParameters`: includes only a handful of features
  and can be used for quick tests. The features which have the "minimal" attribute are used here.
* :class:`tsfresh.feature_extraction.settings.EfficientFCParameters`: Mostly the same features as in the
  :class:`tsfresh.feature_extraction.settings.ComprehensiveFCParameters`, but without features which are marked with the
  "high_comp_cost" attribute. This can be used if runtime performance plays a major role.

Theoretically, you could calculate an unlimited number of features with tsfresh by adding entry after entry to the
dictionary.


For the ambitious: How do I set the parameters for different type of time series?
---------------------------------------------------------------------------------

It is also possible to control the features to be extracted for the different kinds of time series individually.
You can do so by passing another dictionary to the extract function as a

  kind_to_fc_parameters = {"kind" : fc_parameters}

parameter. This dict must be a mapping from kind names (as string) to `fc_parameters` objects,
which you would normally pass as an argument to the `default_fc_parameters` parameter.

So, for example the following code snippet:

.. code:: python

    kind_to_fc_parameters = {
        "temperature": {"mean": None},
        "pressure": {"maximum": None, "minimum": None}
    }

will extract the `"mean"` feature of the `"temperature"` time series and the `"minimum"` and `"maximum"` of the
`"pressure"` time series.

The `kind_to_fc_parameters` argument will partly override the `default_fc_parameters`. So, if you include a kind
name in the `kind_to_fc_parameters` parameter, its value will be used for that kind.
Other kinds will still use the `default_fc_parameters`.


A handy trick: Do I really have to create the dictionary by hand?
-----------------------------------------------------------------

Not necessarily. Let's assume you have a DataFrame of tsfresh features.
By using feature selection algorithms you find out that only a subgroup of features is relevant.


Then, we provide the :func:`tsfresh.feature_extraction.settings.from_columns` method that constructs the `kind_to_fc_parameters`
dictionary from the column names of this filtered feature matrix to make sure that only relevant features are extracted.

This can save a huge amount of time because you prevent the calculation of unnecessary features.
Let's illustrate this with an example:

.. code:: python

    # X_tsfresh contains the extracted tsfresh features
    X_tsfresh = extract_features(...)

    # which are now filtered to only contain relevant features
    X_tsfresh_filtered = some_feature_selection(X_tsfresh, y, ....)

    # we can easily construct the corresponding settings object
    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(X_tsfresh_filtered)

The above code will construct for you the `kind_to_fc_parameters` dictionary that corresponds to the features and parameters (!) from
the tsfresh features that were filtered by the `some_feature_selection` feature selection algorithm.
