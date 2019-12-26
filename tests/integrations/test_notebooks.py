# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import os
import subprocess
import tempfile
import nbformat

from unittest import TestCase

default_timeout = 900


def _notebook_run(path, timeout=default_timeout):
    """
    Execute a singular ipython notebook at path and returns the cells and outputs

    :returns (parsed nb object, execution errors)
    """

    dirname, _ = os.path.split(path)
    execproc_timeout = '--ExecutePreprocessor.timeout=%d' % timeout

    # Do not run notebook tests on Travis.  notebooks tests should only be
    # run in the local developer testing context and notebook tests often
    # cause time out failures on Travis builds see (github #409, #410)
    try:
        if os.environ['TRAVIS']:
            return [], []
    except BaseException:
        pass

    # Ensure temporary files are not auto-deleted as processes have limited
    # permissions to re-use file handles under WinNT-based operating systems.
    fname = ''
    with tempfile.NamedTemporaryFile(mode='w+t', suffix=".ipynb", delete=False) as fout:
        fname = fout.name

        args = ["jupyter", "nbconvert",
                "--to", "notebook", "--execute", execproc_timeout]
        args += ["--ExecutePreprocessor.kernel_name=python3"]
        args += ["--output", fout.name, path]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)
    os.remove(fname)

    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]
    return nb, errors


class NotebooksTestCase(TestCase):

    def test_basic_pipeline_example(self):
        nb, errors = _notebook_run('notebooks/basic_pipeline_example.ipynb', default_timeout)
        self.assertEqual(errors, [])

    def test_friedrich_coefficients(self):
        nb, errors = _notebook_run('notebooks/friedrich_coefficients.ipynb', default_timeout)
        self.assertEqual(errors, [])

    def test_human_activity_recognition_multi_class_example(self):
        nb, errors = _notebook_run('notebooks/human_activity_recognition_multi_class_example.ipynb', default_timeout)
        self.assertEqual(errors, [])

    def test_inspect_dft_features(self):
        nb, errors = _notebook_run('notebooks/inspect_dft_features.ipynb', default_timeout)
        self.assertEqual(errors, [])

    def test_pipeline_with_two_datasets(self):
        nb, errors = _notebook_run('notebooks/pipeline_with_two_datasets.ipynb', default_timeout)
        self.assertEqual(errors, [])

    def test_robot_failure_example(self):
        nb, errors = _notebook_run('notebooks/robot_failure_example.ipynb', default_timeout)
        self.assertEqual(errors, [])

    def test_perform_PCA_on_extracted_features(self):
        nb, errors = _notebook_run('notebooks/perform-PCA-on-extracted-features.ipynb', default_timeout)
        self.assertEqual(errors, [])

    def test_fc_parameters_extraction_dictionary(self):
        nb, errors = _notebook_run('notebooks/the-fc_parameters-extraction-dictionary.ipynb', default_timeout)
        self.assertEqual(errors, [])

    def test_timeseries_forecasting_basic_example(self):
        nb, errors = _notebook_run('notebooks/timeseries_forecasting_basic_example.ipynb', default_timeout)
        self.assertEqual(errors, [])

    def test_timeseries_forecasting_google_stock(self):
        nb, errors = _notebook_run('notebooks/timeseries_forecasting_google_stock.ipynb', default_timeout)
        self.assertEqual(errors, [])

    def test_visualize_benjamini_yekutieli_procedure(self):
        nb, errors = _notebook_run('notebooks/visualize-benjamini-yekutieli-procedure.ipynb', default_timeout)
        self.assertEqual(errors, [])

    def test_feature_extraction_with_datetime_index(self):
        nb, errors = _notebook_run('notebooks/feature_extraction_with_datetime_index.ipynb', default_timeout)
        self.assertEqual(errors, [])
