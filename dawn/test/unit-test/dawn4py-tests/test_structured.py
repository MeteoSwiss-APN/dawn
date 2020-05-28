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

import utils


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
    for backend in (
        dawn4py.CodeGenBackend.CXXNaive,
        dawn4py.CodeGenBackend.GridTools,
        dawn4py.CodeGenBackend.CUDA,
    ):
        dawn4py.compile(sir, backend=backend)
        dawn4py.codegen(
            dawn4py.optimize(
                dawn4py.lower_and_optimize(sir, groups=[]),
                groups=[
                    dawn4py.PassGroup.SetStageName,
                    dawn4py.PassGroup.StageReordering,
                    # dawn4py.PassGroup.StageMerger,
                    dawn4py.PassGroup.SetCaches,
                    dawn4py.PassGroup.SetBlockSize,
                ],
            ),
            backend=backend,
        )
        # TODO There was not test here...
