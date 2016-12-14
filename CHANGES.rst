=========
Changelog
=========

tsfresh uses `Semantic Versioning <http://semver.org/>`_


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
- added feature caculator sampleentropy
- added minimalfeaturesignificance extraction settings
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
