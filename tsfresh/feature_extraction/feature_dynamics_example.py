import dask.dataframe as dd
from extraction import extract_feature_dynamics
from tsfresh.feature_extraction.settings import (
    MinimalFCParameters,
    EfficientFCParameters,
)
from tsfresh.feature_selection import select_features

from tsfresh.feature_extraction.derive_features_dictionaries import (
    derive_features_dictionaries,
)

from tsfresh.feature_extraction.gen_pdf_for_feature_dynamics import (
    gen_pdf_for_feature_dynamics,
)
from tsfresh.feature_extraction.engineer_input_timeseries import (
    engineer_input_timeseries,
)

from tsfresh.feature_extraction.gen_example_timeseries_data import (
    gen_example_timeseries_data,
)

# NOTE: The intent of this file is NOT to be a test suite but more of a "debug playground"


def controller(
    run_dask,
    run_pandas,
    run_efficient,
    run_minimal,
    run_select,
    run_extract_on_selected,
    engineer_more_ts,
    run_pdf,
):

    assert (
        run_dask + run_pandas < 2 and run_dask + run_pandas > 0
    ), "select one of run_dask and run_pandas"
    if run_dask:
        container_type = "dask"
    elif run_pandas:
        container_type = "pandas"

    assert (
        run_efficient + run_minimal < 2 and run_efficient + run_minimal > 0
    ), "select one of run_efficient and run_minimal"
    if run_efficient:
        # Ignore time-based feature calculators "linear_trend_timewise"
        sub_default_fc_parameters = EfficientFCParameters()
        default_fc_parameters = EfficientFCParameters()
    elif run_minimal:
        sub_default_fc_parameters = MinimalFCParameters()
        default_fc_parameters = MinimalFCParameters()

    # run_extract_on_selected ----> run_select
    assert (
        not (run_extract_on_selected) or run_select
    ), "must select features if you want to extract on selected features"

    config_dict = {
        "Container": container_type,
        "Feature Calculators": {
            "Feature Timeseries": sub_default_fc_parameters,
            "Feature Dynamics": default_fc_parameters,
        },
        "Select": run_select,
        "Extract On Selected": run_extract_on_selected,
        "Engineer More Timeseries": engineer_more_ts,
        "Explain Features with pdf": run_pdf,
    }

    return config_dict


##############################################################################################

if __name__ == "__main__":

    ###############################
    ###############################
    # Control variables here
    run_dask = False
    run_pandas = True
    run_efficient = True
    run_minimal = False
    run_select = True
    run_extract_on_selected = True
    engineer_more_ts = False
    run_pdf = True
    ###############################
    ###############################

    # Set up config
    config = controller(
        run_dask,
        run_pandas,
        run_efficient,
        run_minimal,
        run_select,
        run_extract_on_selected,
        engineer_more_ts,
        run_pdf,
    )

    # generate the data
    container_type = "dask" if run_dask else "pandas"
    ts, response = gen_example_timeseries_data(container_type=container_type)

    # Engineer some input timeseries
    if engineer_more_ts:
        if run_dask:
            ts = ts.compute()

        ts = engineer_input_timeseries(
            timeseries=ts,
            column_sort="t",
            column_id="measurement_id",
            compute_differences_within_series=True,
            compute_differences_between_series=True,
        )

        # Include second order differences:.
        ts = ts.merge(
            engineer_input_timeseries(timeseries=ts[["dt_y1", "dt_y2", "dt_y3"]])
        )
        print(ts)

        if run_dask:
            # turn pandas back to dask after engineering more input timeseries
            ts = dd.from_pandas(ts, npartitions=3)

    print(f"\nTime series input:\n\n{ts}")
    print(f"\nTime series response vector:\n\n{response}")
    window_length = 3
    X = extract_feature_dynamics(
        timeseries_container=ts,
        window_length=window_length,
        n_jobs=0,
        feature_timeseries_fc_parameters=config["Feature Calculators"][
            "Feature Timeseries"
        ],
        feature_dynamics_fc_parameters=config["Feature Calculators"][
            "Feature Dynamics"
        ],
        column_id="measurement_id",
        column_sort="t",
        column_kind=None,
        column_value=None,
        show_warnings=False,
    )
    print(f"\nFeature dynamics matrix:\n\n{X}")

    if config["Select"]:
        # select_features does not support dask dataframes
        if config["Container"] == "dask":
            X = X.compute()

        X_filtered = select_features(X, response, fdr_level=0.95)

        # Now get names of the features
        rel_feature_names = list(X_filtered.columns)
        print(f"\nRelevant feature names:\n\n{rel_feature_names}")

        # Now generate a dictionary(s) to extract JUST these features
        feature_time_series_dict, feature_dynamics_dict = derive_features_dictionaries(
            rel_feature_names
        )

        # interpret feature dynamics
        if config["Explain Features with pdf"]:
            subset_of_rel_feature_names = rel_feature_names[0:5]
            gen_pdf_for_feature_dynamics(
                feature_dynamics_names=subset_of_rel_feature_names,
                window_length=window_length,
            )

        if config["Extract On Selected"]:
            X = extract_feature_dynamics(
                timeseries_container=ts,
                window_length=window_length,
                n_jobs=0,
                feature_timeseries_kind_to_fc_parameters=feature_time_series_dict,
                feature_dynamics_kind_to_fc_parameters=feature_dynamics_dict,
                column_id="measurement_id",
                column_sort="t",
                column_kind=None,
                column_value=None,
                show_warnings=False,
            )
            print(f"Relevant Feature Dynamics Matrix{X}")

        print("Success")
