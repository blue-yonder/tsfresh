# a file that has the algorithm used in my project...

# main features on features framework used in VSB project
def features_on_features_vsb(ts,
                             first_fc_params,
                             second_fc_params,
                             fc_params_is_kind,
                             replacement_token):
    '''
    main algorithm that uses tsfresh which computes features on features (feature dynamics) for the VSB data
    NOTE: The DATA format that this function supports is input option 1 for feature extraction https://tsfresh.readthedocs.io/en/latest/text/data_formats.html

        params:
            ts (pd.DataFrame): The VSB measurements that will be processed.
            first_fc_params, second_fc_params (dictionary): feature sets for (1) the extraction of feature time series and (2) the extraction of feature dynamics
            fc_params_is_kind (bool): if the feature dictionaries maps each separate VSB signal (ts kind) to a different feature set then this value is True, otherwise false.
            replacement_token (str): token that replaces double unscore in feature naming convention. This adjustment is required for featue dynamics extraction
        returns:
            X1 (pd.DataFrame): Feature dynamics matrix
            y1 (pd.Series): Response vector.
    '''

    # map the reponse variable to each row in the features on features matrix if the timeseries is test data otherwise do nothing...
    try:
        y = ts.groupby("measurement_id").last()["response"]
    except:
        y = None


    # assign unique pairs of (mes_id, window_id) to each element
    ts["column_id"] = ts["measurement_id"].astype(str) + ", " + ts["window_id"].astype(str)


    # drop the columns which are not relevant to feature extraction
    try:
        ts = ts.drop(columns = ["measurement_id", "window_id", "response"]) # for labelled data
    except:
        ts = ts.drop(columns = ["measurement_id", "window_id"]) # non labelled data

    print("TS INPUT {}".format(ts))
    # first round of feature extraction FEATURE TIME SERIES
    X0 = (extract_features(ts, column_id = "column_id",column_sort = "time_index", kind_to_fc_parameters = first_fc_params, disable_progressbar = True) if fc_params_is_kind
          else extract_features(ts, column_id = "column_id",column_sort = "time_index", default_fc_parameters = first_fc_params, disable_progressbar = True))

    print("FIRST {}".format(X0.shape))

    # drop any features that produce any NaNs/NAs
    if X0.isnull().values.any():
        # store dropped features
        dropped_feature_names = [col_name for col_name in X0.columns[X0.isna().any()].tolist()]
        # store the feature calculators that fail in a file that is constantly updated.
        with open("dropped_feature_names.txt", "a") as f:
            for feature in dropped_feature_names: f.write(feature[feature.index("__") + 2:] + "\n") # 2 is a magic number. It works. But this should be refactored..

        print("found " + str(len(X0.columns[X0.isna().any()].tolist())) + " features from the set of " + str(len(X0.columns)) + " features which should be dropped before being input into second feat extraction")
        X0 = X0.dropna(axis = "columns")


    # tsfresh cant handle double underscores twice so change this in preparation for the second feature extraction
    X0.columns = [str(col_name).replace("__",replacement_token) for col_name in X0.columns]

    # assign windows as the original measurment ID... i.e. extracting "mes_id" from (mes_id, window_id)
    X0["column_id"] = X0.index.to_series().str.split(", ", expand = True).iloc[:,0]

    print("FEATURE TS INPUT {}".format(X0))

    # second round of feature extraction FEATURE DYNAMICS
    X1 = (extract_features(X0, column_id = "column_id", kind_to_fc_parameters = second_fc_params, disable_progressbar = True) if fc_params_is_kind
          else  extract_features(X0, column_id = "column_id", default_fc_parameters = second_fc_params, disable_progressbar = True))

    X1.index.name = "measurement_id"

    # drop any features which are null or na
    if X1.isnull().values.any():
        print("found " + str(len(X1.columns[X1.isna().any()].tolist())) + " features from the set of " + str(len(X1.columns)) + " features which should be dropped before being considered as the final output...")
        X1 = X1.dropna(axis = "columns")


    # sort column names
    X1.sort_index(axis="columns", inplace=True)

    print("X1 output {}, {}".format(X1.shape, X1))

    # returning the feature matrix, the response variable corresponding to each feature matrix window, and optionally the dropped colnames
    return (X1, y)





### The code written into tsfresh

class IterableTsData(Iterable[Timeseries], Sized, TsData):
    """
    Special class of TsData, which can be partitioned.
    Derived classes should implement __iter__ and __len__.
    """
    def pivot(self, results):
        """
        Helper function to turn an iterable of tuples with three entries into a dataframe.

        The input ``list_of_tuples`` needs to be an iterable with tuples containing three
        entries: (a, b, c).
        Out of this, a pandas dataframe will be created with all a's as index,
        all b's as columns and all c's as values.

        It basically does a pd.pivot(first entry, second entry, third entry),
        but optimized for non-pandas input (= python list of tuples).

        This function is called in the end of the extract_features call.
        """
        return_df_dict = defaultdict(dict)
        for chunk_id, variable, value in results:
            # we turn it into a nested mapping `column -> index -> value`
            return_df_dict[variable][chunk_id] = value

        # the mapping column -> {index -> value}
        # is now a dict of dicts. The pandas dataframe
        # constructor will peel this off:
        # first, the keys of the outer dict (the column)
        # will turn into a column header and the rest into a column
        # the rest is {index -> value} which will be turned into a
        # column with index.
        # All index will be aligned.
        return_df = pd.DataFrame(return_df_dict, dtype=float)

        # copy the type of the index
        return_df.index = return_df.index.astype(self.df_id_type)

        # Sort by index to be backward compatible
        return_df = return_df.sort_index()

        return return_df

    def __len__(self):
        """Override in a subclass"""
        raise NotImplementedError

    def __iter__(self):
        """Override in a subclass"""
        raise NotImplementedError


class ApplyableTsData(TsData):
    """
    TsData base class to use, if an iterable ts data can not be used.
    Its only interface is an apply function, which should be applied
    to each of the chunks of the data. How this is done
    depends on the implementation.
    """
    def apply(self, f, **kwargs):
        raise NotImplementedError

def extract_features_on_sub_features(timeseries_container,
                                     sub_feature_split,
                                     sub_default_fc_parameters=None, sub_kind_to_fc_parameters=None,
                                     default_fc_parameters=None, kind_to_fc_parameters=None,
                                     column_id=None, column_sort=None, column_kind=None, column_value=None,
                                     **kwargs):
    ts_data = to_tsdata(timeseries_container, column_id=column_id, column_sort=column_sort, column_kind=column_kind, column_value=column_value)
    if isinstance(ts_data, Iterable):
        split_ts_data = IterableSplitTsData(ts_data, sub_feature_split)
    else:
        split_ts_data = ApplyableSplitTsData(ts_data, sub_feature_split)

    sub_features = extract_features(split_ts_data, default_fc_parameters=sub_default_fc_parameters,
                                    kind_to_fc_parameters=sub_kind_to_fc_parameters, **kwargs, pivot=False)

    column_kind = column_kind or "variable"
    column_id = column_id or "id"
    column_sort = column_sort or "sort"
    column_value = column_value or "value"

    # The feature names include many "_", which will confuse tsfresh where the sub feature name ends
    # and where the real feature name starts. We just remove them.
    # Also, we split up the index into the id and the sort
    # We need to do this separately for dask dataframes,
    # as the return type is not a list, but already a dataframe
    if isinstance(sub_features, dd.DataFrame):
        sub_features = sub_features.reset_index(drop=True)

        sub_features[column_kind] = sub_features[column_kind].apply(lambda col: col.replace("_", ""), meta=(column_kind, object))

        sub_features[column_sort] = sub_features[column_id].apply(lambda x: x[1], meta=(column_id, "int64"))
        sub_features[column_id] = sub_features[column_id].apply(lambda x: x[0], meta=(column_id, ts_data.df_id_type))

    else:
        sub_features = pd.DataFrame(sub_features, columns=[column_id, column_kind, column_value])

        sub_features[column_kind] = sub_features[column_kind].apply(lambda col: col.replace("_", ""))

        sub_features[column_sort] = sub_features[column_id].apply(lambda x: x[1])
        sub_features[column_id] = sub_features[column_id].apply(lambda x: x[0])

    X = extract_features(sub_features, column_id=column_id, column_sort=column_sort, column_kind=column_kind, column_value=column_value,
                         default_fc_parameters=default_fc_parameters, kind_to_fc_parameters=kind_to_fc_parameters,
                         **kwargs)

    return X
