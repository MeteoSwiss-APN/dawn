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

import io

import pytest

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils

from . import utils


@pytest.mark.parametrize("name", utils.GRID_TEST_CASES)
def test_sir_serialization(name):
    sir = getattr(utils, f"make_{name}_sir")(name=name)

    serialized_bytes = dawn4py.serialization.to_bytes(sir)
    assert serialized_bytes is not None
    sir_from_bytes = dawn4py.serialization.from_bytes(serialized_bytes, SIR.SIR)
    assert sir == sir_from_bytes

    serialized_json = dawn4py.serialization.to_json(sir)
    assert serialized_json is not None
    sir_from_json = dawn4py.serialization.from_json(serialized_json, SIR.SIR)
    assert sir == sir_from_json

    bytes_strio = io.StringIO()
    sir_utils.pprint(sir_from_bytes, file=bytes_strio)
    json_strio = io.StringIO()
    sir_utils.pprint(sir_from_json, file=json_strio)

    assert bytes_strio.getvalue() == json_strio.getvalue()


def test_compilation(grid_sir_with_reference_code):
    sir, reference_code = grid_sir_with_reference_code
    backend = "cuda"

    unit_info = {}
    code = dawn4py.compile(sir, backend=backend, unit_info=unit_info)
    # with open("new_code.hpp", "w") as f:
    #     f.write(code)
    # assert code == reference_code
    assert {"filename", "pp_defines", "stencils", "globals"} == set(unit_info.keys())

    unit_info = {}
    code = dawn4py.compile(sir, backend=backend, unit_info=unit_info)
    # assert code == reference_code
    assert {"filename", "pp_defines", "stencils", "globals"} == set(unit_info.keys())

    unit_info = {}
    code = dawn4py.compile(sir, backend=backend, unit_info=unit_info)
    # assert code == reference_code
    assert {"filename", "pp_defines", "stencils", "globals"} == set(unit_info.keys())
