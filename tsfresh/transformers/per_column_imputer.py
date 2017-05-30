# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh.utilities.dataframe_functions import get_range_values_per_column, impute_dataframe_range
import pandas as pd

class PerColumnImputer(BaseEstimator, TransformerMixin):
        """
        Sklearn-compatible estimator, for column-wise imputing DataFrames by replacing all ``NaNs`` and ``infs``
        with with average/extreme values from the same columns. It is basically a wrapper around
        :func:`~tsfresh.utilities.dataframe_functions.impute`.

        Each occurring ``inf`` or ``NaN`` in the DataFrame is replaced by

        * ``-inf`` -> ``min``
        * ``+inf`` -> ``max``
        * ``NaN`` -> ``median``

        This estimator - as most of the sklearn estimators - works in a two step procedure. First, the .fit
        function is called where for each column the min, max and median are computed.
        Secondly, the .transform function is called which replaces the occurances of ``NaNs`` and ``infs`` using
        the column-wise computed min, max and median values.
        """
        def __init__(self, col_to_NINF_repl=None, col_to_PINF_repl=None, col_to_NAN_repl=None):
            """
            Create a new PerColumnImputer instance, optionally with dictionaries containing replacements for
            ``NaNs`` and ``infs``.
            """
            self.col_to_NINF_repl = col_to_NINF_repl
            self.col_to_PINF_repl = col_to_PINF_repl
            self.col_to_NAN_repl = col_to_NAN_repl

        def fit(self, X, y=None):
            """
            Compute the min, max and median for all columns in the DataFrame. For more information,
            please see the :func:`~tsfresh.utilities.dataframe_functions.get_range_values_per_column` function.

            :param X: DataFrame to calculate min, max and median values on
            :type X: pandas.DataFrame

            :return: the estimator with the computed min, max and median values
            :rtype: Imputer
            """
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X.copy())

            if self.col_to_NINF_repl is None or self.col_to_PINF_repl is None or self.col_to_NAN_repl is None:
                col_to_max, col_to_min, col_to_median = get_range_values_per_column(X)
                if self.col_to_NINF_repl is None:
                    self.col_to_NINF_repl = col_to_min
                if self.col_to_PINF_repl is None:
                    self.col_to_PINF_repl = col_to_max
                if self.col_to_NAN_repl is None:
                    self.col_to_NAN_repl = col_to_median

            return self

        def transform(self, X):
            """
            Column-wise replace all ``NaNs``, ``-inf`` and ``+inf`` in the DataFrame `X` with average/extreme
            values from the provided dictionaries.

            :param X: DataFrame to impute
            :type X: pandas.DataFrame

            :return: imputed DataFrame
            :rtype: pandas.DataFrame
            :raise RuntimeError: if the dictionaries are not filled and thus the .fit function has not been called.
            """

            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X.copy())

            if self.col_to_NINF_repl is None or self.col_to_PINF_repl is None or self.col_to_NAN_repl is None:
                raise RuntimeError("PerColumnImputer is not fitted")

            X = impute_dataframe_range(X, self.col_to_PINF_repl, self.col_to_NINF_repl, self.col_to_NAN_repl)

            return X