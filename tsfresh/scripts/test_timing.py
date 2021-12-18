# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2017
import time
from glob import glob
from subprocess import check_output

import pandas as pd


def simulate_with_length(length, df):
    from tsfresh import extract_features

    start = time.time()
    df = extract_features(df[:length], column_id="id", column_sort="time")
    end = time.time()

    duration = end - start

    print("Some checks with length", length)
    print(100 * duration)
    print(len(df.columns), len(df))
    print(
        df[
            [
                "a__abs_energy",
                "b__absolute_sum_of_changes",
                "f__time_reversal_asymmetry_statistic__lag_1",
            ]
        ].head()
    )

    return {"length": length, "duration": duration}


def plot_results():
    from matplotlib import pyplot as plt

    plt.figure(figsize=(7, 7))

    baseline = (
        pd.read_csv("a57a09fe62a62fe0d2564a056f7fd99f58822312.dat")
        .groupby("length")
        .duration.mean()
    )

    for file_name in glob("*.dat"):
        df = pd.read_csv(file_name).groupby("length").duration.mean()

        plt.subplot(211)
        df.plot(label=file_name.replace(".dat", ""))

        plt.subplot(212)
        (baseline / df).plot(label=file_name.replace(".dat", ""))

    plt.subplot(211)
    plt.xlabel("DataFrame Length")
    plt.ylabel("Extract Features Mean Duration")
    plt.legend()

    plt.subplot(212)
    plt.xlabel("DataFrame Length")
    plt.ylabel("Speedup")
    plt.gca().axhline(1, color="black", ls="--")
    plt.legend()

    plt.savefig("timing.png")


def measure_temporal_complexity():
    from tsfresh.examples.robot_execution_failures import (
        download_robot_execution_failures,
        load_robot_execution_failures,
    )

    download_robot_execution_failures()
    df, y = load_robot_execution_failures()

    commit_hash = (
        check_output(["git", "log", '--format="%H"', "-1"])
        .decode("ascii")
        .strip()
        .replace('"', "")
    )

    lengths_to_test = [1, 5, 10, 60, 100, 400, 600, 1000, 2000]
    results = []

    for length in lengths_to_test:
        results.append(simulate_with_length(length, df))
        results.append(simulate_with_length(length, df))
        results.append(simulate_with_length(length, df))

    results = pd.DataFrame(results)
    results.to_csv("{hash}.dat".format(hash=commit_hash))


if __name__ == "__main__":
    measure_temporal_complexity()
    plot_results()
