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
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils

OUTPUT_NAME = "tridiagonal_solve_stencil"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"


def main(args: argparse.Namespace):

    # ---- First vertical region statement ----
    interval_1 = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)
    body_ast_1 = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("c"),
                sir_utils.make_binary_operator(
                    sir_utils.make_field_access_expr("c"), "/", sir_utils.make_field_access_expr("b"),
                ),
                "=",
            )
        ]
    )

    vertical_region_stmt_1 = sir_utils.make_vertical_region_decl_stmt(
        body_ast_1, interval_1, SIR.VerticalRegion.Forward
    )

    # ---- Second vertical region statement ----
    interval_2 = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 1, 0)

    body_ast_2 = sir_utils.make_ast(
        [
            sir_utils.make_var_decl_stmt(
                sir_utils.make_type(SIR.BuiltinType.Integer),
                "m",
                0,
                "=",
                sir_utils.make_expr(
                    sir_utils.make_binary_operator(
                        sir_utils.make_literal_access_expr("1.0", SIR.BuiltinType.Float),
                        "/",
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr("b"),
                            "-",
                            sir_utils.make_binary_operator(
                                sir_utils.make_field_access_expr("a"),
                                "*",
                                sir_utils.make_field_access_expr("c", [0, 0, -1]),
                            ),
                        ),
                    )
                ),
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("c"),
                sir_utils.make_binary_operator(
                    sir_utils.make_field_access_expr("c"), "*", sir_utils.make_var_access_expr("m")
                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("d"),
                sir_utils.make_binary_operator(
                    sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("d"),
                        "-",
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr("a"),
                            "*",
                            sir_utils.make_field_access_expr("d", [0, 0, -1]),
                        ),
                    ),
                    "*",
                    sir_utils.make_var_access_expr("m"),
                ),
                "=",
            ),
        ]
    )
    vertical_region_stmt_2 = sir_utils.make_vertical_region_decl_stmt(
        body_ast_2, interval_2, SIR.VerticalRegion.Forward
    )

    # ---- Third vertical region statement ----
    interval_3 = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, -1)
    body_ast_3 = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("d"),
                sir_utils.make_binary_operator(
                    sir_utils.make_field_access_expr("c"), "*", sir_utils.make_field_access_expr("d", [0, 0, 1]),
                ),
                "-=",
            )
        ]
    )

    vertical_region_stmt_3 = sir_utils.make_vertical_region_decl_stmt(
        body_ast_3, interval_3, SIR.VerticalRegion.Backward
    )

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        SIR.GridType.Value("Cartesian"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast([vertical_region_stmt_1, vertical_region_stmt_2, vertical_region_stmt_3]),
                [
                    sir_utils.make_field("a", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("b", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("c", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("d", sir_utils.make_field_dimensions_cartesian()),
                ],
            )
        ],
    )

    # print the SIR
    if args.verbose:
        sir_utils.pprint(sir)

    # compile
    code = dawn4py.compile(sir, backend="cuda")

    # write to file
    print(f"Writing generated code to '{OUTPUT_PATH}'")
    with open(OUTPUT_PATH, "w") as f:
        f.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a tridiagonal solve computation stencil using Dawn compiler")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print the generated SIR",
    )
    main(parser.parse_args())
