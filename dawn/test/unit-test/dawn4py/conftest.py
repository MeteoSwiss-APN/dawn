##===-----------------------------------------------------------------------------*- Python -*-===##
##                          _
##                         | |
##                       __| | __ ___      ___ ___
##                      / _` |/ _` \ \ /\ / / '_  |
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT).
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

"""Pytest configurations and fixture definitions."""


import os.path

import pytest

from . import utils


@pytest.fixture(params=utils.GRID_TEST_CASES)
def grid_sir_with_reference_code(request):
    sir = getattr(utils, f"make_{request.param}_sir")(name=request.param)
    with open(
        os.path.join(os.path.dirname(__file__), "data", f"{request.param}_reference.cpp"), "r"
    ) as f:
        reference_code = f.read()

    return sir, reference_code


@pytest.fixture(params=utils.UNSTRUCTURED_TEST_CASES)
def unstructure_sir_with_reference_code(request):
    sir = getattr(utils, f"make_{request.param}_sir")(name=request.param)
    reference_code = ""

    return sir, reference_code
