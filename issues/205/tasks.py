from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, extract_features

import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm
import matplotlib.pylab as plt
import b2luigi as luigi
import json

np.random.seed(42)


class DataCreationTask(luigi.Task):
    num_ids = luigi.IntParameter(default=100)
    time_series_length = luigi.IntParameter()

    def output(self):
        yield self.add_to_output("data.csv")

    def run(self):
        df = pd.concat([
            pd.DataFrame({
                "id": [i] * self.time_series_length,
                "time": range(self.time_series_length),
                "value": np.random.randn(self.time_series_length)
            })
            for i in range(self.num_ids)
        ])

        df.to_csv(self.get_output_file_name("data.csv"))


@luigi.requires(DataCreationTask)
class TimingTask(luigi.Task):
    feature_parameter = luigi.DictParameter(hashed=True)
    n_jobs = luigi.IntParameter()
    try_number = luigi.IntParameter()

    def output(self):
        yield self.add_to_output("result.json")

    def run(self):
        input_file = self.get_input_file_names("data.csv")[0]

        df = pd.read_csv(input_file)

        start_time = time()
        extract_features(df, column_id="id", column_sort="time", n_jobs=self.n_jobs,
                         default_fc_parameters=self.feature_parameter,
                         disable_progressbar=True)
        end_time = time()

        single_parameter_name = list(self.feature_parameter.keys())[0]
        single_parameter_params = self.feature_parameter[single_parameter_name]

        result_json = {
            "time": end_time - start_time,
            "n_ids": self.num_ids,
            "n_jobs": self.n_jobs,
            "feature": single_parameter_name,
            "number_parameters": len(single_parameter_params) if single_parameter_params else 0,
            "time_series_length": int((df["id"] == 0).sum()),
            "try_number": self.try_number,
        }

        with open(self.get_output_file_name("result.json"), "w") as f:
            json.dump(result_json, f)


@luigi.requires(DataCreationTask)
class FullTimingTask(luigi.Task):
    n_jobs = luigi.IntParameter()

    def output(self):
        yield self.add_to_output("result.json")

    def run(self):
        input_file = self.get_input_file_names("data.csv")[0]

        df = pd.read_csv(input_file)

        start_time = time()
        extract_features(df, column_id="id", column_sort="time", n_jobs=self.n_jobs,
                         disable_progressbar=True)
        end_time = time()

        result_json = {
            "time": end_time - start_time,
            "n_ids": self.num_ids,
            "n_jobs": self.n_jobs,
            "time_series_length": int((df["id"] == 0).sum()),
        }

        with open(self.get_output_file_name("result.json"), "w") as f:
            json.dump(result_json, f)


class CombinerTask(luigi.Task):
    def complete(self):
        return False

    def requires(self):
        settings = ComprehensiveFCParameters()
        for job in [0, 1, 4]:
            for time_series_length in [100, 500, 1000, 5000]:
                yield FullTimingTask(time_series_length=time_series_length,
                                     n_jobs=job,
                                     num_ids=10)
                yield FullTimingTask(time_series_length=time_series_length,
                                     n_jobs=job,
                                     num_ids=100)

                for feature_name in settings:
                    yield TimingTask(
                        feature_parameter={feature_name: settings[feature_name]},
                        time_series_length=time_series_length,
                        n_jobs=job,
                        num_ids=100,
                        try_number=0,
                    )

                    for try_number in range(3):
                        yield TimingTask(
                            feature_parameter={feature_name: settings[feature_name]},
                            n_jobs=job,
                            try_number=try_number,
                            num_ids=10,
                            time_series_length=time_series_length
                        )

    def output(self):
        yield self.add_to_output("results.csv")

    def run(self):
        results = []

        for input_file in self.get_input_file_names("result.json"):
            with open(input_file, "r") as f:
                results.append(json.load(f))

        df = pd.DataFrame(results)
        df.to_csv(self.get_output_file_name("results.csv"))


if __name__ == "__main__":
    luigi.set_setting("result_path", "results")
    luigi.process(CombinerTask())