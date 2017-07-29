# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import os
import subprocess
import tempfile
import nbformat
import six

from unittest import TestCase

def _notebook_run(path):
    """
    Execute a singular ipython notebook at path and returns the cells and outputs

    :returns (parsed nb object, execution errors)
    """

    dirname, _ = os.path.split(path)

    with tempfile.NamedTemporaryFile(mode='w+t', suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert",
                "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=600"]
        if six.PY2:
            args += ["--ExecutePreprocessor.kernel_name=python2"]
        elif six.PY3:
            args += ["--ExecutePreprocessor.kernel_name=python3"]
        args += ["--output", fout.name, path]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"] \
              if output.output_type == "error"]
    return nb, errors


class NotebooksTestCase(TestCase):

    def test_basic_pipeline_example(self):
        nb, errors = _notebook_run('notebooks/basic_pipeline_example.ipynb')
        self.assertEqual(errors, [])

    def test_human_activity_recognition_multi_class_example(self):
        nb, errors = _notebook_run('notebooks/human_activity_recognition_multi_class_example.ipynb')
        self.assertEqual(errors, [])

    def test_robot_failure_example(self):
        nb, errors = _notebook_run('notebooks/robot_failure_example.ipynb')
        self.assertEqual(errors, [])

    def test_inspect_dft_features(self):
        nb, errors = _notebook_run('notebooks/inspect_dft_features.ipynb')
        self.assertEqual(errors, [])

    def test_fc_parameters_extraction_dictionary(self):
        nb, errors = _notebook_run('notebooks/the-fc_parameters-extraction-dictionary.ipynb')
        self.assertEqual(errors, [])

    def test_pipeline_with_two_datasets(self):
        nb, errors = _notebook_run('notebooks/pipeline_with_two_datasets.ipynb')
        self.assertEqual(errors, [])

    def test_friedrich_coefficients(self):
        nb, errors = _notebook_run('notebooks/friedrich_coefficients.ipynb')
        self.assertEqual(errors, [])