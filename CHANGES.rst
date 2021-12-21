=========
Changelog
=========

tsfresh uses `Semantic Versioning <http://semver.org/>`_

Version 0.19.0
==============

- Breaking Change
    - Drop Python 3.6 support due to dependency on statsmodels 0.13

- Added Features
    - Improve documentation (#831, #834, #851, #853, #870)
    - Add absolute_maximum and mean_n_absolute_max features (#833)
    - Make settings pickable (#845, #847, #910)
    - Disable multiprocessing for `n_jobs=1` (#852)
    - Add black, isort, and pre-commit (#876)

- Bugfixes/Typos/Documentation:
    - Fix conversion of time-series into sequence for lempel_ziv_complexity (#806)
    - Fix range count config (#827)
    - Reword documentation (#893)
    - Fix statsmodels deprecation issues (#898, #912)
    - Fix typo in requirements (#903)
    - Bump statsmodels to v0.13 (#
    - Updated references


Version 0.18.0
==============

- Added Features
    - Allow arbitrary rolling sizes (#766)
    - Allow for multiclass significance tests (#762)
    - Add multiclass option to RelevantFeatureAugmenter (#782)
    - Addition of matrix_profile feature (#793)
    - Added new query similarity counter feature (#798)
    - Add root mean square feature (#813)
- Bugfixes/Typos/Documentation:
    - Do not send coverage of notebook tests to codecov (#759)
    - Fix typos in notebook (#757, #780)
    - Fix output format of `make_forecasting_frame` (#758)
    - Fix badges and remove benchmark test
    - Fix BY notebook plot (#760)
    - Ts forecast example improvement (#763)
    - Also surpress warnings in dask (#769)
    - Update relevant_feature_augmenter.py (#779)
    - Fix column names in quick_start.rst (#778)
    - Improve relevance table function documentation (#781)
    - Fixed #789 Typo in "how to add custom feature" (#790)
    - Convert to the correct type on warnings (#799)
    - Fix minor typos in the docs (#802)
    - Add unwanted filetypes to gitignore (#819)
    - Fix build and test failures (#815)
    - Fix imputing docu (#800)
    - Bump the scikit-learn version (#822)

Version 0.17.0
==============

We changed the default branch from "master" to "main".

- Breaking Change
    - Changed constructed id in roll_time_series from string to tuple (#700)
    - Same for add_sub_time_series_index (#720)
- Added Features
    - Implemented the Lempel-Ziv-Complexity and the Fourier Entropy (#688)
    - Prevent #524 by adding an assert for common identifiers (#690)
    - Added permutation entropy (#691)
    - Added a logo :-) (#694)
    - Implemented the benford distribution feature (#689)
    - Reworked the notebooks (#701, #704)
    - Speed up the result pivoting (#705)
    - Add a test for the dask bindings (#719)
    - Refactor input data iteration to need less memory (#707)
    - Added benchmark tests (#710)
    - Make dask a possible input format (#736)
- Bugfixes:
    - Fixed a bug in the selection, that caused all regression tasks with un-ordered index to be wrong (#715)
    - Fixed readthedocs (#695, #696)
    - Fix spark and dask after #705 and for non-id named id columns (#712)
    - Fix in the forecasting notebook (#729)
    - Let tsfresh choose the value column if possible (#722)
    - Move from coveralls github action to codecov (#734)
    - Improve speed of data processing (#735)
    - Fix for newer, more strict pandas versions (#737)
    - Fix documentation for feature calculators (#743)

Version 0.16.0
==============

- Breaking Change
    - Fix the sorting of the parameters in the feature names (#656)
      The feature names consist of a sorted list of all parameters now.
      That used to be true for all non-combiner features, and is now also true for combiner features.
      If you relied on the actual feature name, this is a breaking change.
    - Change the id after the rolling (#668)
      Now, the old id of your data is still kept. Additionally, we improved the way
      dataframes without a time column are rolled and how the new sub-time series
      are named.
      Also, the documentation was improved a lot.
- Added Features
    - Added variation coefficient (#654)
    - Added the datetimeindex explanation from the notebook to the docs (#661)
    - Optimize RelevantFeatureAugmenter to avoid re-extraction (#669)
    - Added a function `add_sub_time_series_index` (#666)
    - Added Dockerfile
    - Speed optimizations and speed testing script (#681)
- Bugfixes
    - Increase the extracted `ar` coefficients to the full parameter range. (#662)
    - Documentation fixes (#663, #664, #665)
    - Rewrote the `sample_entropy` feature calculator (#681)
      It is now faster and (hopefully) more correct.
      But your results will change!


Version 0.15.1
==============

- Changelog and documentation fixes

Version 0.15.0
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
