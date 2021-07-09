# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2017
"""
This script can be run with:


.. code-block:: bash

   python run_tsfresh.py path_to_your_csv.csv

A corresponding csv containing time series features will be
saved as features_path_to_your_csv.csv

There are a few limitations though

- Currently this only samples to first 50 values.
- Your csv must be space delimited.
- Output is saved as path_to_your_csv.features.csv

"""

import argparse
import os
import sys

import pandas as pd

from tsfresh import extract_features


def _preprocess(df):
    """
    given a DataFrame where records are stored row-wise, rearrange it
    such that records are stored column-wise.
    """

    df = df.stack()

    df.index.rename(["id", "time"], inplace=True)  # .reset_index()
    df.name = "value"
    df = df.reset_index()

    return df


def main(console_args=None):
    parser = argparse.ArgumentParser(
        description="Extract features from time series stored in a CSV file and "
        "write them back into another CSV file. The time series in the CSV "
        "file should either have one of the dataframe-formats described in "
        "http://tsfresh.readthedocs.io/en/latest/text/data_formats.html, "
        "which means you have to supply the --csv-with-headers flag "
        "or should be in the form "
        "[time series 1 values ..., time series 2 values ...] "
        "where you should not add the --csv-with-headers flag. "
        "The CSV is expected to be space-separated."
    )
    parser.add_argument(
        "input_file_name", help="File name of the input CSV file to read in."
    )
    parser.add_argument(
        "--output-file-name",
        help="File name of the output CSV file to write to. "
        "Defaults to input_file_name.features.csv",
        default=None,
    )

    parser.add_argument(
        "--column-sort",
        help="Column name to be used to sort the rows. "
        "Only available when --csv-with-headers is enabled.",
        default=None,
    )
    parser.add_argument(
        "--column-kind",
        help="Column name where the kind column can be found."
        "Only available when --csv-with-headers is enabled.",
        default=None,
    )
    parser.add_argument(
        "--column-value",
        help="Column name where the values can be found."
        "Only available when --csv-with-headers is enabled.",
        default=None,
    )
    parser.add_argument(
        "--column-id",
        help="Column name where the ids can be found."
        "Only available when --csv-with-headers is enabled.",
        default=None,
    )

    parser.add_argument("--csv-with-headers", action="store_true", help="")
    print(console_args)
    args = parser.parse_args(console_args)

    if (
        args.column_id or args.column_kind or args.column_sort or args.column_value
    ) and (not args.csv_with_headers):
        raise AttributeError(
            "You can only pass in column-value, column-kind, column-id or column-sort if "
            "--csv-with-headers is enabled."
        )

    if args.csv_with_headers:
        column_kind = args.column_kind
        column_sort = args.column_sort
        column_value = args.column_value
        column_id = args.column_id
        header = 0
    else:
        column_kind = None
        column_sort = "time"
        column_value = "value"
        column_id = "id"
        header = None

    # Read in CSV file
    input_file_name = args.input_file_name
    df = pd.read_csv(input_file_name, delim_whitespace=True, header=header)

    if not args.csv_with_headers:
        df = _preprocess(df)

    df_features = extract_features(
        df,
        column_kind=column_kind,
        column_sort=column_sort,
        column_value=column_value,
        column_id=column_id,
    )

    # re-cast index from float to int
    df_features.index = df_features.index.astype("int")

    # write to disk
    default_out_file_name = os.path.splitext(input_file_name)[0] + ".features.csv"
    output_file_name = args.output_file_name or default_out_file_name
    df_features.to_csv(output_file_name)


if __name__ == "__main__":
    main()
