# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import os
from multiprocessing import cpu_count


TEST_FOR_BINARY_TARGET_BINARY_FEATURE = "fisher"
TEST_FOR_BINARY_TARGET_REAL_FEATURE = "mann"
TEST_FOR_REAL_TARGET_BINARY_FEATURE = "mann"
TEST_FOR_REAL_TARGET_REAL_FEATURE = "kendall"

FDR_LEVEL = 0.05
HYPOTHESES_INDEPENDENT = False

WRITE_SELECTION_REPORT = False
RESULT_DIR = "logging"

N_PROCESSES = int(os.getenv("NUMBER_OF_CPUS") or cpu_count())
CHUNKSIZE = None
