=========
Changelog
=========

tsfresh uses `Semantic Versioning <http://semver.org/>`_

Version 0.10.0
=============
- new feature calculators:
    - partial autocorrelation
- added list of calculated features to documentation
- added two ipython notebooks to
    - illustrate PCA on features
    - illustrate the Benjamini Yekutieli procedure
- fixed the following bugs
    - improperly quotation of dickey fuller settings

Version 0.9.0
=============
- new feature calculators:
    - ratio_beyond_r_sigma
    - energy_ratio_by_chunks
    - number_crossing_m
    - c3
    - angle & abs for fft coefficients
    - agg_autocorrelation
    - p-Value and usedLag for augmented_dickey_fuller
    - change_quantiles
- changed the calculation of the following features:
    - fft_coefficients
    - autocorrelation
    - time_reversal_asymmetry_statistic
- removed the following feature calculators:
    - large_number_of_peak
    - mean_autocorrelation
    - mean_abs_change_quantiles
- add support for multi classification in the feature selection
- improved description of the rolling mechanism
- added function make_forecasting_frame method for forecasting tasks
- internally ditched the pandas representation of the time series, yielding drastic speed improvements
- replaced feature calculator types from aggregate/aggregate with parameter/apply to simple/combiner
- add test for the ipython notebooks
- added notebook to inspect dft features
- make sure that RelevantFeatureAugmentor always imputes
- fixed the following bugs
    - impute was replacing whole columns by mean
    - fft coefficient were only calculated on truncated part
    - allow to suppress warnings from impute function
    - added missing lag in time_reversal_asymmetry_statistic

Version 0.8.1
=============
- new features:
    - linear trend
    - agg trend
- new sklearn compatible transformers
    - PerColumnImputer
- fixed bugs
    - make mannwhitneyu method compatible with scipy > v0.18.0
- added caching to travis
- internally, added serial calculation of features

Version 0.8.0
=============
- Breaking API changes:
    - removing of feature extraction settings object, replaced by keyword arguments and a plain dictionary (fc_parameters)
    - removing of feature selection settings object, replaced by keyword arguments
- added notebook with examples of new API
- added chapter in docs about the new API
- adjusted old notebooks and documentation to new API

Version 0.7.1
=============

- added a maximum shift parameter to the rolling utility
- added a FAQ entry about how to use tsfresh on windows
- drastically decreased the runtime of the following features
    - cwt_coefficient
    - index_mass_quantile
    - number_peaks
    - large_standard_deviation
    - symmetry_looking
- removed baseline unit tests
- bugfixes:
    - per sample parallel imputing was done on chunks which gave non deterministic results
    - imputing on dtypes other that float32 did not work properly
- several improvements to documentation

Version 0.7.0
=============

- new rolling utility to use tsfresh for time series forecasting tasks
- bugfixes:
    - index_mass_quantile was using global index of time series container
    - an index with same name as id_column was breaking parallelization
    - friedrich_coefficients and max_langevin_fixed_point were occasionally stalling

Version 0.6.0
=============

- progress bar for feature selection
- new feature: estimation of largest fixed point of deterministic dynamics
- new notebook: demonstration how to use tsfresh in a pipeline with train and test datasets
- remove no logging handler warning
- fixed bug in the RelevantFeatureAugmenter regarding the evaluate_only_added_features parameters

Version 0.5.0
=============

- new example: driftbif simulation
- further improvements of the parallelization
- language improvements in the documentation
- performance improvements for some features
- performance improvements for the impute function
- new feature and feature renaming: sum_of_recurring_values, sum_of_recurring_data_points

Version 0.4.0
=============

- fixed several bugs: checking of UCI dataset, out of index error for mean_abs_change_quantiles
- added a progress bar denoting the progress of the extraction process
- added parallelization per sample
- added unit tests for comparing results of feature extraction to older snapshots
- added "high_comp_cost" attribute
- added ReasonableFeatureExtraction settings only calculating features without "high_comp_cost" attribute

Version 0.3.1
=============

- fixed several bugs: closing multiprocessing pools / index out of range cwt calculator / division by 0 in index_mass_quantile
- now all warnings are disabled by default
- for a singular type time series data, the name of value column is used as feature prefix

Version 0.3.0
=============

- fixed bug with parsing of "NUMBER_OF_CPUS" environment variable
- now features are calculated in parallel for each type

Version 0.2.0
=============

- now p-values are calculated in parallel
- fixed bugs for constant features
- allow time series columns to be named 0
- moved uci repository datasets to github mirror
- added feature calculator sample_entropy
- added MinimalFeatureExtraction settings
- fixed bug in calculation of fourier coefficients

Version 0.1.2
=============

- added support for python 3.5.2
- fixed bug with the naming of the features that made the naming of features non-deterministic

Version 0.1.1
=============

- mainly fixes for the read-the-docs documentation, the pypi readme and so on

Version 0.1.0
=============

- Initial version :)
