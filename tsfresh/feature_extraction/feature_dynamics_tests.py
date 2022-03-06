import pandas as pd
from extraction import extract_features, extract_features_on_sub_features
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters
# Test is going to go here. Should work with Louis code + Scott code + the code that is integrated into tsfresh






if __name__ == "__main__":
    
    # Read in data
    ts = pd.read_csv("./test_data.csv")
    print(ts)

    # # running on minimal
    # X = extract_features(timeseries_container= ts,
    #                     n_jobs = 0,
    #                     default_fc_parameters=MinimalFCParameters(),
    #                     column_id= "measurement_id",
    #                     column_sort = "t",
    #                     show_warnings = True)

    # print(X)

    # X = extract_features_on_sub_features(timeseries_container = ts,
    #                                     sub_feature_split = 3, # window size
    #                                     n_jobs = 0,
    #                                     sub_default_fc_parameters = MinimalFCParameters(),
    #                                     default_fc_parameters = MinimalFCParameters(),
    #                                     column_id = "measurement_id",
    #                                     column_sort = "t",
    #                                     column_kind = None,
    #                                     column_value = None,
    #                                     show_warnings = True)
    # print(X)

    # drop feature calculators that are problematic...


    # # running on efficient
    # X = extract_features(timeseries_container= ts,
    #                     n_jobs = 0,
    #                     default_fc_parameters=EfficientFCParameters(),
    #                     column_id= "measurement_id",
    #                     column_sort = "t",
    #                     show_warnings = True)
    #print(X)

    X = extract_features_on_sub_features(timeseries_container = ts,
                                        sub_feature_split = 3,
                                        n_jobs = 0,
                                        sub_default_fc_parameters = EfficientFCParameters(),
                                        default_fc_parameters = EfficientFCParameters(),
                                        column_id = "measurement_id",
                                        column_sort = "t",
                                        column_kind = None,
                                        column_value = None,
                                        show_warnings = True)

    print(X)