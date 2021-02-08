# -*- coding: utf-8 -*-
##===-----------------------------------------------------------------------------*- Python -*-===##
# _
# | |
# __| | __ ___      ___ ___
# / _` |/ _` \ \ /\ / / '_  |
# | (_| | (_| |\ V  V /| | | |
# \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
# This file is distributed under the MIT License (MIT).
# See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

"""Generate input for StageSplitAllStatements tests"""

import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as serial_utils
from google.protobuf.json_format import MessageToJson, Parse


def make_stencil(outputfile, body_ast):
    interval = serial_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )

    sir = serial_utils.make_sir(
        outputfile,
        SIR.GridType.Value("Unstructured"),
        [
            serial_utils.make_stencil(
                "generated",
                serial_utils.make_ast([vertical_region_stmt]),
                [
                ],
            ),
        ],
    )
    f = open(outputfile, "w")
    f.write(MessageToJson(sir))
    f.close()


if __name__ == "__main__":
    # for the "no stmt" test, use the "one stmt" output and delete the statement
    # (from SIR it is not possible to generate a stage without statements)

    one_stmt = serial_utils.make_ast(
        [
            serial_utils.make_var_decl_stmt(
                serial_utils.make_type(SIR.BuiltinType.Integer),
                "a")
        ]
    )
    make_stencil(
        "../input/test_stage_split_all_statements_one_stmt.sir", one_stmt)

    two_stmts = serial_utils.make_ast(
        [
            serial_utils.make_var_decl_stmt(
                serial_utils.make_type(SIR.BuiltinType.Integer),
                "a"),
            serial_utils.make_var_decl_stmt(
                serial_utils.make_type(SIR.BuiltinType.Integer),
                "b")
        ]
    )
    make_stencil(
        "../input/test_stage_split_all_statements_two_stmt.sir", two_stmts)
