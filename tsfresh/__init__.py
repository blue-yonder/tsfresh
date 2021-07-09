# todo: here should go a top level description (see for example the numpy top level __init__.py)

"""
At the top level we export the three most important submodules of tsfresh, which are:

    * :mod:`~tsfresh.extract_features`
    * :mod:`~tsfresh.select_features`
    * :mod:`~tsfresh.extract_relevant_features`
"""


import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"


# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())


from tsfresh.convenience.relevant_extraction import (  # noqa: E402
    extract_relevant_features,
)
from tsfresh.feature_extraction import extract_features  # noqa: E402
from tsfresh.feature_selection import select_features  # noqa: E402
