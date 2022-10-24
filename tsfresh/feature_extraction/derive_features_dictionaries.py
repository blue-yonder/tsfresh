from tsfresh.feature_extraction.settings import from_columns
from typing import List, Tuple


def derive_features_dictionaries(feature_names: List[str]) -> Tuple[dict, dict]:
    """
    Derives and writes out two feature dictionaries which can be used with the feature dynamics framework.

    Return the dictionaries as a single object, and a flag specifying what type of dictionary... i.e. if it is columns --> feature dict

        params:
            feature_names (list of str): the relevant feature names in the form of <ts_kind>||<feature_time_series>__<feature_dynamic>

        returns:
            feature_timeseries_mapping (dict):
            feature_dynamics_mapping (dict):
    """

    assert bool(feature_names) and all(
        isinstance(feature_name, str) for feature_name in feature_names
    )

    replacement_token = "||"
    feature_dynamics_mapping = from_columns(feature_names)
    feature_timeseries_mapping = from_columns(
        [str(x).replace(replacement_token, "__") for x in [*feature_dynamics_mapping]]
    )
    return feature_timeseries_mapping, feature_dynamics_mapping
