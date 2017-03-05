=========
Changelog
=========

tsfresh uses `Semantic Versioning <http://semver.org/>`_

Version 0.6.0
=============

- progress bar for feature selection
- new feature: estimation of largest fixed point of deterministic dynamics
- new notebook: demonstration how to use tsfresh in a pipeline with train and test datasets
- remove no logging handler warning
- fix bug in the RelevantFeatureAugmenter regarding the evaluate_only_added_features parameterqq

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
