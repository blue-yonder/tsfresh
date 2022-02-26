import pandas as pd
from extraction import extract_features_on_sub_features
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters
# Test is going to go here. Should work with Louis code + Scott code + the code that is integrated into tsfresh






if __name__ == "__main__":
    
    # Read in data
    ts = pd.read_csv("./feature_extraction/test_data.csv")
    print(ts)

    # running on minimal
    X = extract_features_on_sub_features(timeseries_container = ts,
                                        sub_feature_split = 1,
                                        sub_default_fc_parameters = MinimalFCParameters(),
                                        default_fc_parameters = MinimalFCParameters(),
                                        column_id = "measurement_id",
                                        column_sort = "t",
                                        column_kind = None,
                                        column_value = None,
                                        show_warnings = True)
    print(X)


    # Running on efficient
    X = extract_features_on_sub_features(timeseries_container = ts,
                                        sub_feature_split = 1,
                                        sub_default_fc_parameters = EfficientFCParameters(),
                                        default_fc_parameters = EfficientFCParameters(),
                                        column_id = "measurement_id",
                                        column_sort = "t",
                                        column_kind = None,
                                        column_value = None,
                                        show_warnings = True)

    print(X)