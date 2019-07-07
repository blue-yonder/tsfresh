#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Maximilian Christ (maximilianchrist.com), 2017, Siesta Ltd.
import pytest

from tsfresh.feature_selection.benjamini_hochberg_test import benjamini_hochberg_test
import pandas as pd
import numpy as np


@pytest.mark.parametrize("p_value, ind, fdr, expected",
                         [([0, 0, 0], True, 0.10, [True, True, True]),
                          ([0, 0, 0], False, 0.10, [True, True, True]),
                          ([0.1, 0.15, 0.2, 0], True, 0.20, [True, True, True, True]),
                          ([0.1, 0.15, 0.2, 0], False, 0.20, [False, False, False, True]),
                          ([0.1, 0.1, 0.05], True, 0.20, [True, True, True]),
                          ([0.1, 0.1, 0.05], False, 0.20, [False, False, False])],
                         )
def test_benjamini_hochberg_test(p_value, ind, fdr, expected):
    df = pd.DataFrame({"p_value": p_value})
    df_p = benjamini_hochberg_test(df_pvalues=df, hypotheses_independent=ind, fdr_level=fdr)
    df_p = df_p.reindex(df.index)
    result = df_p["relevant"].values
    expected = np.array(expected)
    assert (result == expected).all()
