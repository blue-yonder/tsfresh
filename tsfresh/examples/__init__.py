"""
Module with exemplary data sets to play around with.

See for eample the :ref:`quick-start-label` section on how to use them.
"""
from .driftbif_simulation import load_driftbif
from .har_dataset import download_har_dataset, load_har_classes, load_har_dataset
from .robot_execution_failures import (
    download_robot_execution_failures,
    load_robot_execution_failures,
)
