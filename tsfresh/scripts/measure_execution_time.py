# This script extracts the execution time for
# various different settings of tsfresh
# using different input data
# Attention: it will run for ~half a day
# Do these calculations in a controlled environment
# (e.g. a cloud provider VM)
# You will need to have b2luigi installed.
import json
from time import time

import b2luigi as luigi
import numpy as np
import pandas as pd

from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    MinimalFCParameters,
    extract_features,
)


class DataCreationTask(luigi.Task):
    """Create random data for testing"""

    num_ids = luigi.IntParameter(default=100)
    time_series_length = luigi.IntParameter()
    random_seed = luigi.IntParameter()

    def output(self):
        yield self.add_to_output("data.csv")

    def run(self):
        np.random.seed(self.random_seed)

        df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "id": [i] * self.time_series_length,
                        "time": range(self.time_series_length),
                        "value": np.random.randn(self.time_series_length),
                    }
                )
                for i in range(self.num_ids)
            ]
        )

        with self._get_output_target("data.csv").open("w") as f:
            df.to_csv(f)


@luigi.requires(DataCreationTask)
class TimingTask(luigi.Task):
    """Run tsfresh with the given parameters"""

    feature_parameter = luigi.DictParameter(hashed=True)
    n_jobs = luigi.IntParameter()
    try_number = luigi.IntParameter()

    def output(self):
        yield self.add_to_output("result.json")

    def run(self):
        input_file = self._get_input_targets("data.csv")[0]

        with input_file.open("r") as f:
            df = pd.read_csv(f)

        start_time = time()
        extract_features(
            df,
            column_id="id",
            column_sort="time",
            n_jobs=self.n_jobs,
            default_fc_parameters=self.feature_parameter,
            disable_progressbar=True,
        )
        end_time = time()

        single_parameter_name = list(self.feature_parameter.keys())[0]
        single_parameter_params = self.feature_parameter[single_parameter_name]

        result_json = {
            "time": end_time - start_time,
            "n_ids": self.num_ids,
            "n_jobs": self.n_jobs,
            "feature": single_parameter_name,
            "number_parameters": len(single_parameter_params)
            if single_parameter_params
            else 0,
            "time_series_length": int((df["id"] == 0).sum()),
            "try_number": self.try_number,
        }

        with self._get_output_target("result.json").open("w") as f:
            json.dump(result_json, f)


@luigi.requires(DataCreationTask)
class FullTimingTask(luigi.Task):
    """Run tsfresh with all calculators for comparison"""

    n_jobs = luigi.IntParameter()

    def output(self):
        yield self.add_to_output("result.json")

    def run(self):
        input_file = self._get_input_targets("data.csv")[0]

        with input_file.open("r") as f:
            df = pd.read_csv(f)

        start_time = time()
        extract_features(
            df,
            column_id="id",
            column_sort="time",
            n_jobs=self.n_jobs,
            disable_progressbar=True,
        )
        end_time = time()

        result_json = {
            "time": end_time - start_time,
            "n_ids": self.num_ids,
            "n_jobs": self.n_jobs,
            "time_series_length": int((df["id"] == 0).sum()),
        }

        with self._get_output_target("result.json").open("w") as f:
            json.dump(result_json, f)


class CombinerTask(luigi.Task):
    """Collect all tasks into a single result.csv file"""

    def complete(self):
        return False

    def requires(self):
        settings = ComprehensiveFCParameters()
        for job in [0, 1, 4]:
            for time_series_length in [100, 500, 1000, 5000]:
                yield FullTimingTask(
                    time_series_length=time_series_length,
                    n_jobs=job,
                    num_ids=10,
                    random_seed=42,
                )
                yield FullTimingTask(
                    time_series_length=time_series_length,
                    n_jobs=job,
                    num_ids=100,
                    random_seed=42,
                )

                for feature_name in settings:
                    yield TimingTask(
                        feature_parameter={feature_name: settings[feature_name]},
                        time_series_length=time_series_length,
                        n_jobs=job,
                        num_ids=100,
                        try_number=0,
                        random_seed=42,
                    )

                    for try_number in range(3):
                        yield TimingTask(
                            feature_parameter={feature_name: settings[feature_name]},
                            n_jobs=job,
                            try_number=try_number,
                            num_ids=10,
                            time_series_length=time_series_length,
                            random_seed=42,
                        )

    def output(self):
        yield self.add_to_output("results.csv")

    def run(self):
        results = []

        for input_file in self._get_input_targets("result.json"):
            with input_file.open("r") as f:
                results.append(json.load(f))

        df = pd.DataFrame(results)

        with self._get_output_target("results.csv").open("w") as f:
            df.to_csv(f)


if __name__ == "__main__":
    luigi.set_setting("result_path", "results")
    luigi.process(CombinerTask())
