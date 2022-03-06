import dask.dataframe as dd
import pandas as pd
import numpy as np

from extraction import extract_features, extract_features_on_sub_features
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters
from tsfresh import select_features

# temp place for new function...
from gen_features_dicts_function import derive_features_dictionaries




# Example is going to go here. Should work with Louis code + Scott code + the code that is integrated into tsfresh






if __name__ == "__main__":
    ts_path = "./test_data.csv" 
    # Read in data
    ts = pd.read_csv(ts_path)
    print(ts)

    response = pd.Series(np.array([0,1,0,1,0,1]), index = [1,2,3,4,5,6])    

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
                                        show_warnings = False)

    print(X)
    

    # Now select features...
    X_filtered = select_features(X, response, fdr_level = 0.95)


    # Now get names of the features

    rel_feature_names = X_filtered.columns
    print(rel_feature_names)

    # Now generate a dictionary to extract JUST these features
    feature_time_series_dict, feature_dynamics_dict = derive_features_dictionaries(rel_feature_names)
    print("Success")
    




    # Dask dataframes test - cannot figure out how to drop NaN columns.. TODO: Figure this out
    #ts = dd.read_csv(ts_path)
    #X = extract_features_on_sub_features(timeseries_container = ts,
    #                                    sub_feature_split = 3,
    #                                    n_jobs = 0,
    #                                    sub_default_fc_parameters = EfficientFCParameters(),
    #                                    default_fc_parameters = EfficientFCParameters(),
    #                                    column_id = "measurement_id",
    #                                    column_sort = "t",
    #                                    column_kind = None,
    #                                    column_value = None,
    #                                    show_warnings = False)
    
    #print(X.head(n = 10))
    #print("Fin.")
