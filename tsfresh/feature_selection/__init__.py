"""
The :mod:`~tsfresh.feature_selection` module contains feature selection algorithms.
Those methods were suited to pick the best explaining features out of a massive amount of features.
Often the features have to be picked in situations where one has more features than samples.
Traditional feature selection methods can be not suitable for such situations which is why we propose a p-value based
approach that inspects the significance of the features individually to avoid overfitting and spurious correlations.
"""


from tsfresh.feature_selection.selection import select_features
