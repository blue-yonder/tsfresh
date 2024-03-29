# -*- coding: utf-8 -*-
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import datetime
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

# -- Hack for ReadTheDocs ------------------------------------------------------
# This hack is necessary since RTD does not issue `sphinx-apidoc` before running
# `sphinx-build -b html . _build/html`. See Issue:
# https://github.com/rtfd/readthedocs.org/issues/1139
# DON'T FORGET: Check the box "Install your project inside a virtualenv using
# setup.py install" in the RTD Advanced Settings.
import os

on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    import inspect
    from sphinx.ext.apidoc import main

    __location__ = os.path.join(
        os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe()))
    )

    output_dir = os.path.join(__location__, "../docs/api")
    module_dir = os.path.join(__location__, "../tsfresh")
    cmd_line_template = "sphinx-apidoc -f -o {outputdir} {moduledir}"
    cmd_line = cmd_line_template.format(outputdir=output_dir, moduledir=module_dir)
    main(cmd_line.split(" ")[1:])

# -- General configuration -----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.imgmath",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
now = datetime.datetime.today()
project = "tsfresh"
copyright = "2023-{}, Maximilian Christ et al./ Blue Yonder GmbH".format(now.year)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = ""  # Is set by calling `setup.py docs`
# The full version, including alpha/beta/rc tags.
release = ""  # Is set by calling `setup.py docs`

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "api/tests*"]

# Boolean indicating whether to scan all found documents for autosummary
# directives, and to generate stub pages for each
autosummary_generate = True

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {"style_nav_header_background": "#51b63c"}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
try:
    from tsfresh import __version__ as version
except ImportError:
    pass
else:
    release = version

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# From https://rackerlabs.github.io/docs-rackspace/tools/rtd-tables.html
html_css_files = [
    'theme_override.css',
]

# Output file base name for HTML help builder.
htmlhelp_basename = "tsfresh-doc"
