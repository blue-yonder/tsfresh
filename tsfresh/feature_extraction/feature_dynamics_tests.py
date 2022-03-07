import dask.dataframe as dd
import pandas as pd
import numpy as np
from sympy import true

from extraction import extract_features, extract_features_on_sub_features
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters
from tsfresh import select_features

# temp place for new function...
from gen_features_dicts_function import derive_features_dictionaries


##############################################################################################
# throwaway functions..
def read_ts(ts_path, response_path, container_type):
    if container_type == "dask":
        ts = dd.read_csv(ts_path)
    elif container_type == "pandas":
        ts = pd.read_csv(ts_path)
    return ts, pd.read_csv(response_path).set_index("measurement_id").squeeze()

def controller(run_dask, run_pandas, run_efficient, run_minimal, run_select, run_extract_on_selected):

    assert run_dask + run_pandas < 2 and run_dask + run_pandas > 0, 'select one of run_dask and run_pandas'
    if run_dask:
        container_type = "dask"
    elif run_pandas: 
        container_type = "pandas"

    assert run_efficient + run_minimal < 2 and run_efficient + run_minimal > 0, 'select one of run_efficient and run_minimal'
    if run_efficient:
        sub_default_fc_parameters = EfficientFCParameters()
        default_fc_parameters = EfficientFCParameters()
    elif run_minimal:
        sub_default_fc_parameters = MinimalFCParameters()
        default_fc_parameters = MinimalFCParameters()
    
    # run_extract_on_selected ----> run_select
    assert not(run_extract_on_selected) or run_select, 'must select features if you want to extract on selected features'

    config_dict = {
        "Container": container_type,
        "Feature Calculators":
        {
            "Feature Timeseries":sub_default_fc_parameters,
            "Feature Dynamics":default_fc_parameters
        },
        "Select":run_select,
        "Extract On Selected":run_extract_on_selected}
    
    return config_dict

##############################################################################################

if __name__ == "__main__":
    ###############################
    ###############################
    # Control variables here
    run_dask = True
    run_pandas = False
    run_efficient = False
    run_minimal = True
    run_select = True
    run_extract_on_selected = True
    ts_path = "./test_data.csv" 
    response_path = "./response.csv"
    ###############################
    ###############################
     
    # Set up config
    config = controller(run_dask, run_pandas, run_efficient, run_minimal, run_select, run_extract_on_selected)

    # Read in data
    ts, response = read_ts(ts_path,response_path,config["Container"])
    print(ts)
    print(response)

    # Dask dataframes test - cannot figure out how to drop NaN columns.. TODO: Figure this out
    # NOTE: May be related to the fact that Dask Name is: pivot_table_sum-agg
    # NOTE: The columns of X are "CategoricalIndex"... 
    X = extract_features_on_sub_features(timeseries_container = ts,
                                        sub_feature_split = 3, # window size
                                        n_jobs = 0,
                                        sub_default_fc_parameters = config["Feature Calculators"]["Feature Timeseries"],
                                        default_fc_parameters = config["Feature Calculators"]["Feature Dynamics"],
                                        column_id = "measurement_id",
                                        column_sort = "t",
                                        column_kind = None,
                                        column_value = None,
                                        show_warnings = False)
    print(X)
    
    if config["Select"]:
        # Now select features...

        # select_features does not support dask dataframes
        if config["Container"] == "dask":
            X = X.compute()

        X_filtered = select_features(X, response, fdr_level = 0.95)

        # Now get names of the features
        rel_feature_names = X_filtered.columns
        print(rel_feature_names)
        
        # Now generate a dictionary(s) to extract JUST these features
        feature_time_series_dict, feature_dynamics_dict = derive_features_dictionaries(rel_feature_names)

        if config["Extract On Selected"]:
            X = extract_features_on_sub_features(timeseries_container = ts,
                                                sub_feature_split = 3, # window size
                                                n_jobs = 0,
                                                sub_default_fc_parameters = config["Feature Calculators"]["Feature Timeseries"],
                                                default_fc_parameters = config["Feature Calculators"]["Feature Dynamics"],
                                                column_id = "measurement_id",
                                                column_sort = "t",
                                                column_kind = None,
                                                column_value = None,
                                                show_warnings = False)
            print(X)

       


        print("Success")
