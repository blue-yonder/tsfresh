Feature extraction settings
===========================

In most of the cases - especially when you play around with the data and do data mining -
you probably want to extract all typical features when you call the :func:`tsfresh.extract_features`
function and select only the relevant features later. Then you can call the function
:func:`tsfresh.extract_features` without passing a `default_calculation_settings_mapping` or
`kind_to_calculation_settings_mapping` object, which means you are using the default options (which will use all
feature calculators in this package).

After digging deeper into your data, you maybe want to calculate more of a certain type of feature and less of another
type. So, you need to use custom settings for the feature extractors. To do that with tsfresh you will have to use a
custom settings object, like shown now:

>>> from tsfresh.feature_extraction import FeatureExtractionSettings
>>> settings = FeatureExtractionSettings()
>>> # Set here the options of the settings object as shown in the chapters below
>>> # ...
>>> from tsfresh.feature_extraction import extract_features
>>> extract_features(df, default_calculation_settings_mapping=settings)


The `default_calculation_settings_mapping` is expected to be a dictionary, which maps feature calculator names
(the function names you can find in the :mod:`tsfresh.feature_extraction.feature_calculators` file) to a list
of dictionaries, which are the parameters with which the function will be called (as key value pairs). Each function
parameter combination, that is in this dict will be called during the extraction and will produce a feature.
If the function does not need any parameters, the list value of the corresponding key can be empty.

For example

.. code:: python

    calculation_settings_mapping = {
        "length": None,
        "large_standard_deviation": [{"r": 0.05}, {"r": 0.1}]
    }

will produce three features: one by calling the
:func:`tsfresh.feature_extraction.feature_calculators.length` function without any parameters and two by calling
:func:`tsfresh.feature_extraction.feature_calculators.large_standard_deviation` with `r = 0.05` and `r = 0.1`.

So you can control, which features will be extracted, by adding/removing either keys or parameters from the dict.

For convenience, there are already three dict predefined, which can be used right away:

* :class:`tsfresh.feature_extraction.settings.FeatureExtractionSettings`: includes all features without parameters and
  all features will parameters, with quite some different parameter combinations. This is the default of you do not
  hand in a `default_calculation_settings_mapping` at all.
* :class:`tsfresh.feature_extraction.settings.MinimalFeatureExtractionSettings`: includes only few features
  and can be used for quick tests. The features which have the "minimal" attribute are used here.
* :class:`tsfresh.feature_extraction.settings.ReasonableFeatureExtractionSettings`: Mostly the same features as in the
  :class:`tsfresh.feature_extraction.settings.FeatureExtractionSettings`, except a few exception, which are marked as
  high_comp_cost. This can be used if runtime performance plays a major role.

It is also possible, to control the features to be extracted for the different kinds of time series individually.
You can do so by passing another dictionary to the extract function as a

`kind_to_calculation_settings_mapping` = {"kind" : `calculation_settings_mapping`}

parameter. This dict must be a mapping from kind names (as string) to `calculation_settings_mapping` objects,
which you would normally pass as an argument to the `default_calculation_settings_mapping` parameter.

This dominating behavior of the `kind_to_calculation_settings_mapping` argument works partly. So, if you include a kind
name in the `kind_to_calculation_settings_mapping` parameter, its value will override the
`default_calculation_settings_mapping`. Otherwise, the `default_calculation_settings_mapping` if the kind name could
not be found.


