from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features

from matplotlib import pyplot as plt
from glob import glob
import time
import pandas as pd
from subprocess import check_output

download_robot_execution_failures()
df, y = load_robot_execution_failures()


def test_with_length(length, df):
    start = time.time()
    df = extract_features(df[:length], column_id='id', column_sort='time')
    end = time.time()

    duration = end - start

    print("Some checks with length", length)
    print(100 * duration)
    print(len(df.columns), len(df))
    print(df[["a__abs_energy", "b__absolute_sum_of_changes", "f__time_reversal_asymmetry_statistic__lag_1"]].head())

    return {"length": length, "duration": duration}


def plot_results():
    plt.figure(figsize=(7, 7))

    baseline = None
    test = None

    for file_name in glob("*.dat"):
        df = pd.read_csv(file_name).groupby("length").duration

        plt.subplot(211)
        df.mean().plot(label=file_name.replace(".dat", ""))

        if "ae2d1121f57a1686478a3f7d4b59ae8c735ce883.dat" in file_name:
            baseline = df.mean()
        else:
            test = df.mean()

    plt.subplot(211)
    plt.xlabel("DataFrame Length")
    plt.ylabel("Extract Features Mean Duration")
    plt.legend()

    plt.subplot(212)
    (test / baseline).plot()
    plt.xlabel("DataFrame Length")
    plt.ylabel("Speedup")
    plt.gca().axhline(1, color="black", ls="--")

    plt.savefig("timing.png")


if __name__ == "__main__":
    commit_hash = check_output(["git", "log", "--format=\"%H\"", "-1"]).decode("ascii").strip().replace("\"", "")

    lengths_to_test = [1, 5, 10, 60, 100, 400, 600, 1000, 2000]
    results = []

    for length in lengths_to_test:
        results.append(test_with_length(length, df))
        results.append(test_with_length(length, df))
        results.append(test_with_length(length, df))

    results = pd.DataFrame(results)
    results.to_csv("{hash}.dat".format(hash=commit_hash))

    plot_results()