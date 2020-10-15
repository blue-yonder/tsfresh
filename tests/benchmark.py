# Testcase for benchmarking tsfresh feature extraction and selection
import pytest
import pandas as pd
import numpy as np
from functools import partial

from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, MinimalFCParameters


def create_data(time_series_length, num_ids, random_seed=42):
    np.random.seed(random_seed)

    df = pd.concat([
        pd.DataFrame({
            "id": [i] * time_series_length,
            "time": range(time_series_length),
            "value": np.random.randn(time_series_length).cumsum()
        })
        for i in range(num_ids)
    ])

    return df


def test_benchmark_small_data(benchmark):
    df = create_data(5, 200)

    benchmark(extract_features, df, column_id="id", column_sort="time", n_jobs=0,
              disable_progressbar=True)


def test_benchmark_large_data(benchmark):
    df = create_data(500, 20)

    benchmark(extract_features, df, column_id="id", column_sort="time", n_jobs=0,
              disable_progressbar=True)


def test_benchmark_with_selection(benchmark):
    df = create_data(500, 20)
    y = pd.Series(np.random.choice([0, 1], 20))

    benchmark(extract_relevant_features, df, y, column_id="id", column_sort="time", n_jobs=0,
              disable_progressbar=True)

def test_optimized_sample_entropy(benchmark):
    fc_parameters = {
        "optimized_sample_entropy": []
    }
    extract_features_partial = partial(extract_features, default_fc_parameters=fc_parameters)
    df = create_data(300, 20)

    benchmark(extract_features_partial, df, column_id="id", column_sort="time", n_jobs=0,
              disable_progressbar=True)

def test_sample_entropy(benchmark):
    fc_parameters = {
        "sample_entropy": []
    }
    extract_features_partial = partial(extract_features, default_fc_parameters=fc_parameters)
    df = create_data(300, 20)

    benchmark(extract_features_partial, df, column_id="id", column_sort="time", n_jobs=0,
              disable_progressbar=True)

def test_sample_entropy_similarity():

    fc_parameters = {
        "sample_entropy": []
    }

    df = create_data(100, 20)

    resnon_opti = extract_features(df, column_id="id", column_sort="time", n_jobs=0,
                                    disable_progressbar=True, default_fc_parameters=fc_parameters)
                                
    fc_parameters = {
        "optimized_sample_entropy": []
    }

    res_opti = extract_features(df, column_id="id", column_sort="time", n_jobs=0,
                                    disable_progressbar=True, default_fc_parameters=fc_parameters)
    

    resnon_opti.columns = res_opti.columns

    pd.testing.assert_frame_equal(resnon_opti, res_opti)