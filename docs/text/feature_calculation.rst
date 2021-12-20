.. _feature-naming-label:

Feature Calculator Naming
=========================

tsfresh enforces a strict naming of the created features, which you have to follow whenever you create new feature
calculators.
This is due to the :func:`tsfresh.feature_extraction.settings.from_columns` method which needs to
deduce the following information from the feature name:

    * the time series that was used to calculate the feature
    * the feature calculator method that was used to derive the feature
    * all parameters that have been used to calculate the feature (optional)

Hence, to enable the :func:`tsfresh.feature_extraction.settings.from_columns` to deduce all the
necessary conditions, the features should be named in the following format:

    {time_series_name}__{feature_name}__{parameter name 1}_{parameter value 1}__[..]__{parameter name k}_{parameter value k}

Here, we assumed that {feature_name} has k parameters.

Examples of feature naming
''''''''''''''''''''''''''

So for example the following feature name:

    temperature_1__quantile__q_0.6

is the value of the feature :func:`tsfresh.feature_extraction.feature_calculators.quantile` for the time series
```temperature_1``` and a parameter value of ``q=0.6``. On the other hand, the feature named:

    Pressure 5__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_14__w_5

denotes the value of the feature :func:`tsfresh.feature_extraction.feature_calculators.cwt_coefficients` for
the time series ```Pressure 5``` under parameter values of ``widths=(2, 5, 10, 20)``, ``coeff=14`` and ``w=5``.
