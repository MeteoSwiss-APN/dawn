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

"""Tridiagonal solve computation SIR generator

This program creates the SIR corresponding to a tridiagonal solve computation using the SIR serialization Python API.
The tridiagonal solve is a basic example that contains vertical data dependencies that need to be resolved
by the compiler passes.
The code is meant as an example for high-level DSLs that could generate SIR from their own
internal IR.
The program contains two parts:
    1. construct the SIR of the example
    2. pass the SIR to the dawn compiler in order to run all optimizer passes and code generation.
       In this example the compiler is configured with the CUDA backend, therefore will code generate
       an optimized CUDA implementation.

"""


import argparse
import os

import dawn4py
from dawn4py.serialization import SIR, AST
from dawn4py.serialization import utils as serial_utils
from google.protobuf.json_format import MessageToJson, Parse

OUTPUT_NAME = "tridiagonal_solve_stencil"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"


def main(args: argparse.Namespace):

    # ---- First vertical region statement ----
    interval_1 = serial_utils.make_interval(AST.Interval.Start, AST.Interval.End, 0, 0)
    body_ast_1 = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("c"),
                serial_utils.make_binary_operator(
                    serial_utils.make_field_access_expr("c"),
                    "/",
                    serial_utils.make_field_access_expr("b"),
                ),
                "=",
            )
        ]
    )

    vertical_region_stmt_1 = serial_utils.make_vertical_region_decl_stmt(
        body_ast_1, interval_1, AST.VerticalRegion.Forward
    )

    # ---- Second vertical region statement ----
    interval_2 = serial_utils.make_interval(AST.Interval.Start, AST.Interval.End, 1, 0)

    body_ast_2 = serial_utils.make_ast(
        [
            serial_utils.make_var_decl_stmt(
                serial_utils.make_type(AST.BuiltinType.Integer),
                "m",
                0,
                "=",
                serial_utils.make_expr(
                    serial_utils.make_binary_operator(
                        serial_utils.make_literal_access_expr("1.0", AST.BuiltinType.Float),
                        "/",
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr("b"),
                            "-",
                            serial_utils.make_binary_operator(
                                serial_utils.make_field_access_expr("a"),
                                "*",
                                serial_utils.make_field_access_expr("c", [0, 0, -1]),
                            ),
                        ),
                    )
                ),
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("c"),
                serial_utils.make_binary_operator(
                    serial_utils.make_field_access_expr("c"), "*", serial_utils.make_var_access_expr("m")
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("d"),
                serial_utils.make_binary_operator(
                    serial_utils.make_binary_operator(
                        serial_utils.make_field_access_expr("d"),
                        "-",
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr("a"),
                            "*",
                            serial_utils.make_field_access_expr("d", [0, 0, -1]),
                        ),
                    ),
                    "*",
                    serial_utils.make_var_access_expr("m"),
                ),
                "=",
            ),
        ]
    )
    vertical_region_stmt_2 = serial_utils.make_vertical_region_decl_stmt(
        body_ast_2, interval_2, AST.VerticalRegion.Forward
    )

    # ---- Third vertical region statement ----
    interval_3 = serial_utils.make_interval(AST.Interval.Start, AST.Interval.End, 0, -1)
    body_ast_3 = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("d"),
                serial_utils.make_binary_operator(
                    serial_utils.make_field_access_expr("c"),
                    "*",
                    serial_utils.make_field_access_expr("d", [0, 0, 1]),
                ),
                "-=",
            )
        ]
    )

    vertical_region_stmt_3 = serial_utils.make_vertical_region_decl_stmt(
        body_ast_3, interval_3, AST.VerticalRegion.Backward
    )

    sir = serial_utils.make_sir(
        OUTPUT_FILE,
        AST.GridType.Value("Cartesian"),
        [
            serial_utils.make_stencil(
                OUTPUT_NAME,
                serial_utils.make_ast(
                    [vertical_region_stmt_1, vertical_region_stmt_2, vertical_region_stmt_3]
                ),
                [
                    serial_utils.make_field("a", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field("b", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field("c", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field("d", serial_utils.make_field_dimensions_cartesian()),
                ],
            )
        ],
    )

    # print the SIR       
    if args.verbose:
        print(MessageToJson(sir))

    # compile
    pass_groups = dawn4py.default_pass_groups()
    pass_groups.insert(1, dawn4py.PassGroup.MultiStageMerger)
    code = dawn4py.compile(sir, groups=pass_groups, backend=dawn4py.CodeGenBackend.CUDA)

    # write to file
    print(f"Writing generated code to '{OUTPUT_PATH}'")
    with open(OUTPUT_PATH, "w") as f:
        f.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a tridiagonal solve computation stencil using Dawn compiler"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print the generated SIR",
    )
    main(parser.parse_args())
