#!/usr/bin/env python

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

import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils
from google.protobuf.json_format import MessageToJson

stencil_name = "hori_diff_stencil"
output_file = f"{stencil_name}.cpp"


if __name__ == "__main__":
    interval = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)

    # create the stencil body AST
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("lap"),
                sir_utils.make_binary_operator(
                    sir_utils.make_binary_operator(
                        sir_utils.make_literal_access_expr("-4.0", SIR.BuiltinType.Float),
                        "*",
                        sir_utils.make_field_access_expr("in"),
                    ),
                    "+",
                    sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("coeff"),
                        "*",
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr("in", [1, 0, 0]),
                            "+",
                            sir_utils.make_binary_operator(
                                sir_utils.make_field_access_expr("in", [-1, 0, 0]),
                                "+",
                                sir_utils.make_binary_operator(
                                    sir_utils.make_field_access_expr("in", [0, 1, 0]),
                                    "+",
                                    sir_utils.make_field_access_expr("in", [0, -1, 0]),
                                ),
                            ),
                        ),
                    ),
                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out"),
                sir_utils.make_binary_operator(
                    sir_utils.make_binary_operator(
                        sir_utils.make_literal_access_expr("-4.0", SIR.BuiltinType.Float),
                        "*",
                        sir_utils.make_field_access_expr("lap"),
                    ),
                    "+",
                    sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("coeff"),
                        "*",
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr("lap", [1, 0, 0]),
                            "+",
                            sir_utils.make_binary_operator(
                                sir_utils.make_field_access_expr("lap", [-1, 0, 0]),
                                "+",
                                sir_utils.make_binary_operator(
                                    sir_utils.make_field_access_expr("lap", [0, 1, 0]),
                                    "+",
                                    sir_utils.make_field_access_expr("lap", [0, -1, 0]),
                                ),
                            ),
                        ),
                    ),
                ),
                "=",
            ),
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(body_ast, interval, SIR.VerticalRegion.Forward)

    sir = sir_utils.make_sir(
        output_file,
        SIR.GridType.Value("Cartesian"),
        [
            sir_utils.make_stencil(
                stencil_name,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field("in", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("out", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("coeff", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("lap", sir_utils.make_field_dimensions_cartesian(), is_temporary=True),
                ],
            )
        ],
    )

    dawn4py.compile(sir, backend="cuda", serialize_iir=True, output_file=output_file)
    os.rename(stencil_name + ".0.iir", stencil_name + ".iir")

