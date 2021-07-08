# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests


@pytest.mark.parametrize(
    "p_value, ind, fdr, expected",
    [
        ([0, 0, 0], True, 0.10, [True, True, True]),
        ([0, 0, 0], False, 0.10, [True, True, True]),
        ([0.1, 0.15, 0.2, 0], True, 0.20, [True, True, True, True]),
        ([0.1, 0.15, 0.2, 0], False, 0.20, [False, False, False, True]),
        ([0.1, 0.1, 0.05], True, 0.20, [True, True, True]),
        ([0.1, 0.11, 0.05], False, 0.20, [False, False, False]),
        ([0.1, 0.1, 0.05], False, 0.20, [True, True, True]),
        (
            [0.00356, 0.01042, 0.01208, 0.02155, 0.03329, 0.11542],
            True,
            0.05,
            [True, True, True, True, True, False],
        ),
        (
            [0.00356, 0.01042, 0.01208, 0.02155, 0.03329, 0.11542],
            False,
            0.05,
            [False, False, False, False, False, False],
        ),
        ([0.11, 0.001, 0.05], False, 0.20, [False, True, True]),
    ],
)
def test_fdr_control(p_value, ind, fdr, expected):
    df = pd.DataFrame({"p_value": p_value})
    method = "fdr_bh" if ind else "fdr_by"
    df["relevant"] = multipletests(pvals=df.p_value, alpha=fdr, method=method)[0]
    result = df["relevant"].values
    expected = np.array(expected)
    assert (result == expected).all()
