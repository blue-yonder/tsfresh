# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import os
import subprocess
import tempfile
from unittest import TestCase

import nbformat
import pytest

default_timeout = 900


def _notebook_run(path, timeout=default_timeout):
    """
    Execute a singular ipython notebook at path and returns the cells and outputs

    :returns (parsed nb object, execution errors)
    """

    dirname, _ = os.path.split(path)
    execproc_timeout = "--ExecutePreprocessor.timeout=%d" % timeout

    # Do not run notebook tests on Travis.  notebooks tests should only be
    # run in the local developer testing context and notebook tests often
    # cause time out failures on Travis builds see (github #409, #410)
    try:
        if os.environ["TRAVIS"]:
            return [], []
    except BaseException:
        pass

    # Ensure temporary files are not auto-deleted as processes have limited
    # permissions to re-use file handles under WinNT-based operating systems.
    fname = ""
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".ipynb", delete=False) as fout:
        fname = fout.name

        args = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            execproc_timeout,
        ]
        args += ["--ExecutePreprocessor.kernel_name=python3"]
        args += ["--output", fout.name, path]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)
    os.remove(fname)

    errors = [
        output
        for cell in nb.cells
        if "outputs" in cell
        for output in cell["outputs"]
        if output.output_type == "error"
    ]
    return nb, errors


@pytest.mark.skipif(
    os.environ.get("TEST_NOTEBOOKS") != "y", reason="Disabled notebook testing"
)
class NotebooksTestCase(TestCase):
    def test_basic_example(self):
        nb, errors = _notebook_run(
            "notebooks/examples/01 Feature Extraction and Selection.ipynb",
            default_timeout,
        )
        self.assertEqual(errors, [])

    def test_pipeline_example(self):
        nb, errors = _notebook_run(
            "notebooks/examples/02 sklearn Pipeline.ipynb", default_timeout
        )
        self.assertEqual(errors, [])

    def test_extraction_settings(self):
        nb, errors = _notebook_run(
            "notebooks/examples/03 Feature Extraction Settings.ipynb", default_timeout
        )
        self.assertEqual(errors, [])

    def test_multiclass_selection_example(self):
        nb, errors = _notebook_run(
            "notebooks/examples/04 Multiclass Selection Example.ipynb", default_timeout
        )
        self.assertEqual(errors, [])

    def test_timeseries_forecasting(self):
        nb, errors = _notebook_run(
            "notebooks/examples/05 Timeseries Forecasting.ipynb", default_timeout
        )
        self.assertEqual(errors, [])

    def test_timeseries_forecasting_exprt(self):
        nb, errors = _notebook_run(
            "notebooks/advanced/05 Timeseries Forecasting (multiple ids).ipynb",
            default_timeout,
        )
        self.assertEqual(errors, [])

    def test_inspect_dft_features(self):
        nb, errors = _notebook_run(
            "notebooks/advanced/inspect_dft_features.ipynb", default_timeout
        )
        self.assertEqual(errors, [])

    def test_feature_extraction_with_datetime_index(self):
        nb, errors = _notebook_run(
            "notebooks/advanced/feature_extraction_with_datetime_index.ipynb",
            default_timeout,
        )
        self.assertEqual(errors, [])

    def test_friedrich_coefficients(self):
        nb, errors = _notebook_run(
            "notebooks/advanced/friedrich_coefficients.ipynb", default_timeout
        )
        self.assertEqual(errors, [])

    def test_inspect_dft_features(self):
        nb, errors = _notebook_run(
            "notebooks/advanced/inspect_dft_features.ipynb", default_timeout
        )
        self.assertEqual(errors, [])

    def test_perform_PCA_on_extracted_features(self):
        nb, errors = _notebook_run(
            "notebooks/advanced/perform-PCA-on-extracted-features.ipynb",
            default_timeout,
        )
        self.assertEqual(errors, [])

    def test_visualize_benjamini_yekutieli_procedure(self):
        nb, errors = _notebook_run(
            "notebooks/advanced/visualize-benjamini-yekutieli-procedure.ipynb",
            default_timeout,
        )
        self.assertEqual(errors, [])
