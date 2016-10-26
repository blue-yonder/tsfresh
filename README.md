# tsfresh

This repository contains the *TSFRESH* python package. The abbreviation stands for

*"Time Series Feature extraction based on scalable hypothesis tests"*.

The package contains many feature extraction methods and a robust feature selection algorithm.

### Spend less time on feature engineering

Data Scientists often spend most of their time either cleaning data or building features.
While we cannot change the first thing, the second can be automated.
*TSFRESH* frees your time spend on building features by extracting them automatically.
Hence, you have more time to study the newest deep learning paper, read hacker news or build better models.


### Automatic extraction of 100s of features

*TSFRESH* automatically extracts 100s of features from time series.
Those features describe basic characteristics of the time series such as the number of peaks, the average or maximal value or more complex features such as the time reversal symmetry statistic.

![The features extracted from a exemplary time series](docs/images/introduction_ts_exa_features.png)

The set of features can then be used to construct statistical or machine learning models on the time series to be used for example in regression or
classification tasks.

### Forget irrelevant features

Time series often contain noise, redundancies or irrelevant information.
As a result most of the extracted features will not be useful for the machine learning task at hand.

To avoid extracting irrelevant features, the *TSFRESH* package has a built-in filtering procedure.
This filtering procedure evaluates the explaining power and importance of each characteristic for the regression or classification tasks at hand.

It is based on the well developed theory of hypothesis testing and uses a multiple test procedure.
As a result the filtering process mathematically controls the percentage of irrelevant extracted features.

The algorithm is described in the following paper

Christ, M., Kempa-Liehr, A.W. and Feindt, M. (2016).
*Distributed and parallel time series feature extraction for industrial big data applications.*
ArXiv e-prints: 1610.07717 URL: [link](http://adsabs.harvard.edu/abs/2016arXiv161007717C)


### Advantages of tsfresh

*TSFRESH* has several selling points, for example

1. it is field tested
2. it is unit tested
3. the filtering process is statistically/mathematically correct
4. it has a comprehensive documentation
5. it is compatible with sklearn, pandas and numpy
6. it allows anyone to easily add his own favorite features

### Futher Reading

If you are interested in the technical workings, go to see our comprehensive Read-The-Docs documentation at [link]().

The algorithm, especially the filtering part are also described in the paper mentioned above.
