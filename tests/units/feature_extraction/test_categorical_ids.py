"""Feature extraction must not invent time series for unused ID categories."""

import pandas as pd
import pytest

from tsfresh import extract_features


@pytest.mark.parametrize("data_format", ["wide", "long", "dict"])
@pytest.mark.parametrize("column_sort", [None, "time"])
@pytest.mark.parametrize("id_type", ["integer", "categorical", "unused_category"])
def test_extract_features_only_observed_ids(data_format, column_sort, id_type):
    """Filtering categorical IDs must produce the same rows as integer IDs."""
    df = pd.DataFrame({"id": [1, 3, 1, 3], "time": [2, 2, 1, 1], "value": [4, 8, 2, 6]})
    if id_type != "integer":
        categories = [1, 2, 3] if id_type == "unused_category" else [1, 3]
        df["id"] = pd.Categorical(df["id"], categories=categories)

    kwargs = {}
    if data_format == "dict":
        data = {"value": df}
        kwargs["column_value"] = "value"
    elif data_format == "long":
        data = df.assign(kind="value")
        kwargs.update(column_kind="kind", column_value="value")
    else:
        data = df
        kwargs["column_value"] = "value"

    result = extract_features(
        data,
        column_id="id",
        column_sort=column_sort,
        default_fc_parameters={"length": None, "sum_values": None},
        n_jobs=1,
        disable_progressbar=True,
        **kwargs,
    )
    expected = pd.DataFrame(
        {"value__length": [2.0, 2.0], "value__sum_values": [6.0, 14.0]},
        index=pd.Index([1, 3], dtype=df["id"].dtype),
    )
    pd.testing.assert_frame_equal(result, expected)
