#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for tsfresh.

    This file was generated with PyScaffold 2.5.6, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

import sys

from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [line for line in f if not line.startswith("#")]

needs_sphinx = {"build_sphinx", "upload_docs"}.intersection(sys.argv)
sphinx = ["sphinx", "sphinx_rtd_theme"] if needs_sphinx else []

setup(
    use_scm_version=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=["setuptools_scm"] + sphinx,
    packages=find_packages(exclude=["tests.*", "tests"]),
    install_requires=requirements,
)
