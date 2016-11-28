|Documentation Status| |Build Status| |Coverage Status| |license|
|Gitter chat| |py27 status| |py352 status|

tsfresh
=======

This repository contains the *TSFRESH* python package. The abbreviation
stands for

*"Time Series Feature extraction based on scalable hypothesis tests"*.

The package contains many feature extraction methods and a robust
feature selection algorithm.

Spend less time on feature engineering
--------------------------------------

Data Scientists often spend most of their time either cleaning data or
building features. While we cannot change the first thing, the second
can be automated. *TSFRESH* frees your time spend on building features
by extracting them automatically. Hence, you have more time to study the
newest deep learning paper, read hacker news or build better models.

Automatic extraction of 100s of features
----------------------------------------

*TSFRESH* automatically extracts 100s of features from time series.
Those features describe basic characteristics of the time series such as
the number of peaks, the average or maximal value or more complex
features such as the time reversal symmetry statistic.

.. figure:: docs/images/introduction_ts_exa_features.png
   :alt: The features extracted from a exemplary time series

   The features extracted from a exemplary time series

The set of features can then be used to construct statistical or machine
learning models on the time series to be used for example in regression
or classification tasks.

Forget irrelevant features
--------------------------

Time series often contain noise, redundancies or irrelevant information.
As a result most of the extracted features will not be useful for the
machine learning task at hand.

To avoid extracting irrelevant features, the *TSFRESH* package has a
built-in filtering procedure. This filtering procedure evaluates the
explaining power and importance of each characteristic for the
regression or classification tasks at hand.

It is based on the well developed theory of hypothesis testing and uses
a multiple test procedure. As a result the filtering process
mathematically controls the percentage of irrelevant extracted features.

The algorithm is described in the following paper

-  Christ, M., Kempa-Liehr, A.W. and Feindt, M. (2016).
   *Distributed and parallel time series feature extraction for
   industrial big data applications.*
   ArXiv e-print 1610.07717, https://arxiv.org/abs/1610.07717.

Advantages of tsfresh
---------------------

*TSFRESH* has several selling points, for example

1. it is field tested
2. it is unit tested
3. the filtering process is statistically/mathematically correct
4. it has a comprehensive documentation
5. it is compatible with sklearn, pandas and numpy
6. it allows anyone to easily add their favorite features

Next steps
----------

If you are interested in the technical workings, go to see our
comprehensive Read-The-Docs documentation at
http://tsfresh.readthedocs.io.

The algorithm, especially the filtering part are also described in the
paper mentioned above.

If you have some questions or feedback you can find the developers in
the `gitter
chatroom. <https://gitter.im/tsfresh/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link>`__

We appreciate any contributions, if you are interested in helping us to
make *TSFRESH* the biggest archive of feature extraction methods in
python, just head over to our
`How-To-Contribute <http://tsfresh.readthedocs.io/en/latest/text/how_to_contribute.html>`__
instructions.

.. |Documentation Status| image:: https://readthedocs.org/projects/tsfresh/badge/?version=latest
   :target: http://tsfresh.readthedocs.io/en/latest/?badge=latest
.. |Build Status| image:: https://travis-ci.org/blue-yonder/tsfresh.svg?branch=master
   :target: https://travis-ci.org/blue-yonder/tsfresh
.. |Coverage Status| image:: https://coveralls.io/repos/github/blue-yonder/tsfresh/badge.svg?branch=master
   :target: https://coveralls.io/github/blue-yonder/tsfresh?branch=master
.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :target: https://github.com/blue-yonder/tsfresh/blob/master/LICENSE.txt
.. |Gitter chat| image:: https://badges.gitter.im/tsfresh/Lobby.svg
   :target: https://gitter.im/tsfresh/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link
.. |py27 status| image:: https://img.shields.io/badge/python2.7-supported-green.svg
.. |py352 status| image:: https://img.shields.io/badge/python3.5.2-supported-green.svg
   :target: https://github.com/blue-yonder/tsfresh/issues/8
