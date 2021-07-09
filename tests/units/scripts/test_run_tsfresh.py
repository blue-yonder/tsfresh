# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
import os
import shutil
import tempfile
from unittest import TestCase

from mock import patch

from tsfresh.scripts import run_tsfresh


class RunTSFreshTestCase(TestCase):
    """
    Test the command line interface to tsfresh. This does not test the tsfresh functionality (this is tested elsewhere),
    but mocks the extract_features functionality by just outputting the same df as going in into the function.
    """

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        # Change into the tmp dir
        self.curr_dir = os.getcwd()
        os.chdir(self.test_dir)

        def extract_features_mock(df, **kwargs):
            # just mirror the input
            return df

        # Patcher object to be disabled in tearDown
        self.patcher = patch(
            "tsfresh.scripts.run_tsfresh.extract_features",
            side_effect=extract_features_mock,
        )
        # Mocked extract function
        self.mocked_extract_function = self.patcher.start()

    def tearDown(self):
        # Remove the directory after the test
        os.chdir(self.curr_dir)
        shutil.rmtree(self.test_dir)
        # Disable the mocking
        self.patcher.stop()

    def call_main_function(self, input_csv_string=None, arguments=""):
        # Helper function to call "main" from the script after creating the input file out of the input string.
        input_file_name = "temporary_csv_file.csv"

        if input_csv_string:
            with open(os.path.join(self.test_dir, input_file_name), "w") as f:
                f.write(input_csv_string)

        output_file_name = "temporary_output_csv_file.csv"
        arguments_with_filenames = "{input_file_name} {arguments} --output-file-name {output_file_name}".format(
            input_file_name=input_file_name,
            arguments=arguments,
            output_file_name=output_file_name,
        )

        run_tsfresh.main(arguments_with_filenames.split())

        if os.path.exists(os.path.join(self.test_dir, output_file_name)):
            with open(os.path.join(self.test_dir, output_file_name), "r") as f:
                return f.read()
        else:
            return None

    def test_invalid_arguments(self):
        self.assertRaises(
            SystemExit, self.call_main_function, arguments="--invalid-argument"
        )

    def test_csv_without_headers_wrong_arguments(self):
        self.assertRaises(
            AttributeError, self.call_main_function, arguments="--column-id invalid"
        )

    def test_csv_without_headers(self):
        input_csv = "1 1 1 1\n1 1 1 1"
        output_csv = (
            ",id,time,value\n"
            "0,0,0,1\n"
            "1,0,1,1\n"
            "2,0,2,1\n"
            "3,0,3,1\n"
            "4,1,0,1\n"
            "5,1,1,1\n"
            "6,1,2,1\n"
            "7,1,3,1\n"
        )

        result_csv = self.call_main_function(input_csv_string=input_csv)

        self.assertEqual(result_csv, output_csv)

        called_kwargs = self.mocked_extract_function.call_args[1]

        self.assertEqual(called_kwargs["column_id"], "id")
        self.assertEqual(called_kwargs["column_value"], "value")
        self.assertIsNone(called_kwargs["column_kind"])
        self.assertEqual(called_kwargs["column_sort"], "time")

    def test_csv_with_header(self):
        input_csv = "ID SORT KIND VALUE\n0 0 a 0\n1 0 a 1\n0 0 b 1\n1 0 b 3"
        output_csv = ",ID,SORT,KIND,VALUE\n0,0,0,a,0\n1,1,0,a,1\n2,0,0,b,1\n3,1,0,b,3\n"

        result_csv = self.call_main_function(
            input_csv_string=input_csv,
            arguments="--column-id ID --column-sort SORT --column-value VALUE --column-kind KIND --csv-with-headers",
        )

        self.assertEqual(result_csv, output_csv)

        called_kwargs = self.mocked_extract_function.call_args[1]

        self.assertEqual(called_kwargs["column_id"], "ID")
        self.assertEqual(called_kwargs["column_value"], "VALUE")
        self.assertEqual(called_kwargs["column_kind"], "KIND")
        self.assertEqual(called_kwargs["column_sort"], "SORT")
