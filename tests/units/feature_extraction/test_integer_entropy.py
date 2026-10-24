import math

import numpy as np
import pandas as pd
import pytest

from tsfresh.feature_extraction.feature_calculators import (
    approximate_entropy,
    sample_entropy,
)


def _template_counts(values, length, tolerance, include_self):
    templates = [values[i : i + length] for i in range(len(values) - length + 1)]
    return [
        sum(
            (include_self or i != j)
            and max(abs(a - b) for a, b in zip(first, second)) <= tolerance
            for j, second in enumerate(templates)
        )
        for i, first in enumerate(templates)
    ]


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int64, np.uint64])
@pytest.mark.parametrize("as_series", [False, True])
def test_entropy_integer_distances(dtype, as_series):
    limits = np.iinfo(dtype)
    values = np.array(
        [limits.min, limits.max, 0, limits.min, limits.max, 1] * 3,
        dtype=dtype,
    )
    original = values.copy()
    # Python integers provide exact, signed template distances, including at
    # the limits of int64/uint64. No NumPy subtraction is used by this oracle.
    integers = [int(value) for value in values]
    tolerance = 0.2 * np.std(values)
    counts2 = _template_counts(integers, 2, tolerance, False)
    counts3 = _template_counts(integers, 3, tolerance, False)
    expected_sample = -math.log(sum(counts3) / sum(counts2))

    def phi(length):
        counts = _template_counts(integers, length, tolerance, True)
        return sum(math.log(count / len(counts)) for count in counts) / len(counts)

    expected_approximate = abs(phi(2) - phi(3))
    signal = pd.Series(values) if as_series else values
    np.testing.assert_allclose(sample_entropy(signal), expected_sample, rtol=1e-14)
    np.testing.assert_allclose(
        approximate_entropy(signal, 2, 0.2), expected_approximate, atol=1e-14
    )
    np.testing.assert_array_equal(values, original)


def test_entropy_boolean_distances():
    signal = np.array([True, False, True, True, False, False] * 3)
    np.testing.assert_allclose(
        sample_entropy(signal), sample_entropy(signal.astype(float))
    )
    np.testing.assert_allclose(
        approximate_entropy(signal, 2, 0.2),
        approximate_entropy(signal.astype(float), 2, 0.2),
    )
