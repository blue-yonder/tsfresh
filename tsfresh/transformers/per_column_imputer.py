# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from tsfresh.utilities.dataframe_functions import (
    get_range_values_per_column,
    impute_dataframe_range,
)


class PerColumnImputer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible estimator, for column-wise imputing DataFrames by replacing all ``NaNs`` and ``infs``
    with with average/extreme values from the same columns. It is basically a wrapper around
    :func:`~tsfresh.utilities.dataframe_functions.impute`.

    Each occurring ``inf`` or ``NaN`` in the DataFrame is replaced by

    * ``-inf`` -> ``min``
    * ``+inf`` -> ``max``
    * ``NaN`` -> ``median``

    This estimator - as most of the sklearn estimators - works in a two step procedure. First, the ``.fit``
    function is called where for each column the min, max and median are computed.
    Secondly, the ``.transform`` function is called which replaces the occurances of ``NaNs`` and ``infs`` using
    the column-wise computed min, max and median values.
    """

    def __init__(
        self,
        col_to_NINF_repl_preset=None,
        col_to_PINF_repl_preset=None,
        col_to_NAN_repl_preset=None,
    ):
        """
        Create a new PerColumnImputer instance, optionally with dictionaries containing replacements for
        ``NaNs`` and ``infs``.

        :param col_to_NINF_repl: Dictionary mapping column names to ``-inf`` replacement values
        :type col_to_NINF_repl: dict
        :param col_to_PINF_repl: Dictionary mapping column names to ``+inf`` replacement values
        :type col_to_PINF_repl: dict
        :param col_to_NAN_repl: Dictionary mapping column names to ``NaN`` replacement values
        :type col_to_NAN_repl: dict
        """
        self._col_to_NINF_repl = None
        self._col_to_PINF_repl = None
        self._col_to_NAN_repl = None
        self.col_to_NINF_repl_preset = col_to_NINF_repl_preset
        self.col_to_PINF_repl_preset = col_to_PINF_repl_preset
        self.col_to_NAN_repl_preset = col_to_NAN_repl_preset

    def fit(self, X, y=None):
        """
        Compute the min, max and median for all columns in the DataFrame. For more information,
        please see the :func:`~tsfresh.utilities.dataframe_functions.get_range_values_per_column` function.

        :param X: DataFrame to calculate min, max and median values on
        :type X: pandas.DataFrame
        :param y: Unneeded.
        :type y: Any

        :return: the estimator with the computed min, max and median values
        :rtype: Imputer
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        col_to_max, col_to_min, col_to_median = get_range_values_per_column(X)

        if self.col_to_NINF_repl_preset is not None:
            if not set(X.columns) >= set(self.col_to_NINF_repl_preset.keys()):
                raise ValueError(
                    "Preset dictionary 'col_to_NINF_repl_preset' contain more keys "
                    "than the column names in X"
                )
            col_to_min.update(self.col_to_NINF_repl_preset)
        self._col_to_NINF_repl = col_to_min

        if self.col_to_PINF_repl_preset is not None:
            if not set(X.columns) >= set(self.col_to_PINF_repl_preset.keys()):
                raise ValueError(
                    "Preset dictionary 'col_to_PINF_repl_preset' contain more keys "
                    "than the column names in X"
                )
            col_to_max.update(self.col_to_PINF_repl_preset)
        self._col_to_PINF_repl = col_to_max

        if self.col_to_NAN_repl_preset is not None:
            if not set(X.columns) >= set(self.col_to_NAN_repl_preset.keys()):
                raise ValueError(
                    "Preset dictionary 'col_to_NAN_repl_preset' contain more keys "
                    "than the column names in X"
                )
            col_to_median.update(self.col_to_NAN_repl_preset)
        self._col_to_NAN_repl = col_to_median

        return self

    def transform(self, X):
        """
        Column-wise replace all ``NaNs``, ``-inf`` and ``+inf`` in the DataFrame `X` with average/extreme
        values from the provided dictionaries.

        :param X: DataFrame to impute
        :type X: pandas.DataFrame

        :return: imputed DataFrame
        :rtype: pandas.DataFrame
        :raise RuntimeError: if the replacement dictionaries are still of None type.
         This can happen if the transformer was not fitted.
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if (
            self._col_to_NINF_repl is None
            or self._col_to_PINF_repl is None
            or self._col_to_NAN_repl is None
        ):
            raise NotFittedError("PerColumnImputer is not fitted")

        X = impute_dataframe_range(
            X, self._col_to_PINF_repl, self._col_to_NINF_repl, self._col_to_NAN_repl
        )

        return X
