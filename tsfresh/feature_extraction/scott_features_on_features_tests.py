import pandas as pd
from extraction import extract_features_on_sub_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
# Test is going to go here. Should work with Louis code + Scott code + the code that is integrated into tsfresh

# Read in data
ts = pd.read_csv("test_data.csv")

print("Minimal: {}".format(MinimalFCParameters()))

X = extract_features_on_sub_features(timeseries_container = ts,
                                     sub_feature_split = 2,
                                     sub_default_fc_parameters = MinimalFCParameters(),
                                     default_fc_parameters = MinimalFCParameters(),
                                     column_id = "measurement_id",
                                     column_sort = "t",
                                     column_kind = None,
                                     column_value = None)

print(X)
#for col in X.columns:
#    print(col)
