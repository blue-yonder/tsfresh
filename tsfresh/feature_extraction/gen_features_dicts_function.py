from tsfresh.feature_extraction.settings import from_columns

def derive_features_dictionaries(feature_names):
    '''
    Derives and writes out two feature dictionaries which can be used with the feature dynamics framework.

    Return the dictionaries as a single object, and a flag specifying what type of dictionary... i.e. if it is columns --> feature dict

        params:
            feature_names (list of str): the relevant feature names in the form of <ts_kind>||<feature_time_series>__<feature_dynamic>

        returns:
            f_mapping (dict):
            f_on_f_mapping (dict):
    '''

    # type check might not be neccessary
    #assert feature_names and all(isinstance(feature_dynamic, str) for feature_dynamic in feature_names)

    replacement_token = "||" # set this as the standard as per the docstring...

    f_on_f_mapping = from_columns(feature_names)
    f_mapping = from_columns([str(x).replace(replacement_token,"__") for x in [*f_on_f_mapping]])

    return f_mapping, f_on_f_mapping
