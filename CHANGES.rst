=========
Changelog
=========

tsfresh uses `Semantic Versioning <http://semver.org/>`_

Version 0.15.1
==============

- Added Features
    - Add count_above and count_below feature (#632)
    - Add convenience bindings for dask dataframes and pyspark dataframes (#651)
- Bugfixes
    - Fix documentation build and feature table in sphinx (#637, #631, #627)
    - Add scripts to API documentation
    - Skip dask test for older python versions (#649)
    - Add missing distributor keyword (#648)
    - Fix tuple input for cwt (#645)

Version 0.14.1
==============

- Fix travis deployment

Version 0.14.0
==============

- Breaking Change
    - Replace Benjamini-Hochberg implementation with statsmodels implementation (#570)
- Refactoring and Documentation
    - travis.yml (#605)
    - gitignore (#608)
    - Fix docstring of c3 (#590)
    - Feature/pep8 (#607)
- Added Features
    - Improve test coverage (#609)
    - Add "autolag" parameter to augmented_dickey_fuller() (#612)
- Bugfixes
    - Feature/pep8 (#607)
    - Fix filtering on warnings with multiprocessing on Windows (#610)
    - Remove outdated logging config (#621)
    - Replace Benjamini-Hochberg implementation with statsmodels implementation (#570)
    - Fix the kernel and the naming of a notebook (#626)


Version 0.13.0
==============

- Drop python 2.7 support (#568)
- Fixed bugs
    - Fix cache in friedrich_coefficients and agg_linear_trend (#593)
    - Added a check for wrong column names and a test for this check (#586)
    - Make sure to not install the tests folder (#599)
    - Make sure there is at least a single column which we can use for data (#589)
    - Avoid division by zero in energy_ratio_by_chunks (#588)
    - Ensure that get_moment() uses float computations (#584)
    - Preserve index when column_value and column_kind not provided (#576)
    - Add @set_property("input", "pd.Series") when needed (#582)
    - Fix off-by-one error in longest strike features (fixes #577) (#578)
    - Add `set_property` import (#572)
    - Fix typo (#571)
    - Fix indexing of melted normalized input (#563)
    - Fix travis (#569)
- Remove warnings (#583)
- Update to newest python version (#594)
- Optimizations
    - Early return from change_quantiles if ql >= qh (#591)
    - Optimize mean_second_derivative_central (#587)
    - Improve performance with Numpy's sum function (#567)
    - Optimize mean_change (fixes issue #542) and correct documentation (#574)


Version 0.12.0
==============

- fixed bugs
    - wrong calculation of friedrich coefficients
    - feature selection selected too many features
    - an ignored max_timeshift parameter in roll_time_series
- add deprecation warning for python 2
- added support for index based features
- new feature calculator
    - linear_trend_timewise
- enable the RelevantFeatureAugmenter to be used in cross validated pipelines
- increased scipy dependency to 1.2.0


Version 0.11.2
==============
- change chunking in energy_ratio_by_chunks to use all data points
- fix warning for spkt_welch_density
- adapt default settings for "value_count" and "range_count"
- added
    - maxlag parameter to agg_autocorrelation function
- now, the kind column of the input DataFrame is cast as str, old derived FC_Settings can become invalid
- only set default_fc_parameters to ComprehensiveFCParameters() if also kind_to_fc_parameters is set None in `extract_features`
- removed pyscaffold
- use asymptotic algorithm to derive kendal tau


Version 0.11.1
==============
- general performance improvements
- removed hard pinning of dependencies
- fixed bugs
    - the stock price forecasting notebook
    - the multi classification notebook

Version 0.11.0
==============
- new feature calculators:
    - fft_aggregated
    - cid_ce
- renamed mean_second_derivate_central to mean_second_derivative_central
- add warning if no relevant features were found in feature selection
- add columns_to_ignore parameter to from_columns method
- add distribution module, contains support for distributed feature extraction on Dask

Version 0.10.1
==============
- split test suite into unit and integration tests
- fixed the following bugs
    - use name of value column as time series kind
    - prevent the spawning of subprocesses which lead to high memory consumption
    - fix deployment from travis to pypi

Version 0.10.0
==============
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
