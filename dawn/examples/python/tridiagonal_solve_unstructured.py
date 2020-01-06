#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ===-----------------------------------------------------------------------------*- Python -*-===##
#                          _
#                         | |
#                       __| | __ ___      ___ ___
#                      / _` |/ _` \ \ /\ / / '_  |
#                     | (_| | (_| |\ V  V /| | | |
#                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
#
#
#  This file is distributed under the MIT License (MIT).
#  See LICENSE.txt for details.
#
# ===------------------------------------------------------------------------------------------===##
"""Tridiagonal solve computation HIR generator

This program creates the HIR corresponding to a tridiagonal solve computation using the Python API of the HIR.
The tridiagonal solve is a basic example that contains vertical data dependencies that need to be resolved
by the compiler passes.
The code is meant as an example for high-level DSLs that could generate HIR from their own
internal IR.
The program contains two parts:
    1. construct the HIR of the example
    2. pass the HIR to the dawn compiler in order to run all optimizer passes and code generation.
       In this example the compiler is configured with the unstrctured naive backend"""

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils

OUTPUT_NAME = "unstructured_vertical_solver"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = os.path.join(os.path.dirname(
    __file__), "data", f"{OUTPUT_NAME}.cpp")


def create_vertical_region_stmt1() -> VerticalRegionDeclStmt:
    """ create a vertical region statement for the stencil
    """

    interval = make_interval(Interval.Start, Interval.Start, 0, 0)

    body_ast = make_ast(
        [make_assignment_stmt(
            make_unstructured_field_access_expr("c"),
            make_binary_operator(
                make_unstructured_field_access_expr("c"),
                "/",
                make_unstructured_field_access_expr("b")
            ),
            "="
        ),
            make_assignment_stmt(
            make_unstructured_field_access_expr("d"),
            make_binary_operator(
                make_unstructured_field_access_expr("d"),
                "/",
                make_unstructured_field_access_expr("b")
            ),
            "="
        )
        ]
    )

    vertical_region_stmt = make_vertical_region_decl_stmt(
        body_ast, interval, VerticalRegion.Forward)
    return vertical_region_stmt


def create_vertical_region_stmt2() -> VerticalRegionDeclStmt:
    """ create a vertical region statement for the stencil
    """

    interval = make_interval(Interval.Start, Interval.End, 1, 0)

    body_ast = make_ast(
        [
            make_var_decl_stmt(
                make_type(BuiltinType.Float),
                "m", 0, "=",
                make_expr(
                    make_binary_operator(
                        make_literal_access_expr("1.0", BuiltinType.Float),
                        "/",
                        make_binary_operator(
                            make_unstructured_field_access_expr("b"),
                            "-",
                            make_binary_operator(
                                make_unstructured_field_access_expr("a"),
                                "*",
                                make_unstructured_field_access_expr(
                                    "c", make_unstructured_offset(False), -1)
                            )
                        )
                    )
                )
            ),
            make_assignment_stmt(
                make_unstructured_field_access_expr("c"),
                make_binary_operator(
                    make_unstructured_field_access_expr("c"),
                    "*",
                    make_var_access_expr("m")
                ),
                "="
            ),
            make_assignment_stmt(
                make_unstructured_field_access_expr("d"),
                make_binary_operator(
                    make_binary_operator(
                        make_unstructured_field_access_expr("d"),
                        "-",
                        make_binary_operator(
                            make_unstructured_field_access_expr("a"),
                            "*",
                            make_unstructured_field_access_expr(
                                "d", make_unstructured_offset(False), - 1)
                        )
                    ),
                    "*",
                    make_var_access_expr("m")
                ),
                "="
            )
        ]
    )

    vertical_region_stmt = make_vertical_region_decl_stmt(
        body_ast, interval, VerticalRegion.Forward)
    return vertical_region_stmt


def create_vertical_region_stmt3() -> VerticalRegionDeclStmt:
    """ create a vertical region statement for the stencil
    """

    interval = make_interval(Interval.Start, Interval.End, 0, -1)

    body_ast = make_ast(
        [make_assignment_stmt(
            make_unstructured_field_access_expr("d"),
            make_binary_operator(
                make_unstructured_field_access_expr("c"),
                "*",
                make_unstructured_field_access_expr(
                    "d", make_unstructured_offset(False), 1)
            ),
            "-="
        )
        ]
    )

    vertical_region_stmt = make_vertical_region_decl_stmt(
        body_ast, interval, VerticalRegion.Backward)
    return vertical_region_stmt


def main(args: argparse.Namespace):
    sir = make_sir(GridType.Value('Triangular'), "unstructured_tridiagonal_solve.cpp", [
        make_stencil(
            "unstructured_tridiagonal_solve",
            make_ast([
                create_vertical_region_stmt1(),
                create_vertical_region_stmt2(),
                create_vertical_region_stmt3()
            ]),
            [make_field("a"), make_field("b"),
             make_field("c"), make_field("d")]
        )

    ])

    # print the SIR
    if args.verbose:
        sir_utils.pprint(sir)

    # compile
    code = dawn4py.compile(sir, backend="c++-naive-ico")

    # write to file
    print(f"Writing generated code to '{OUTPUT_PATH}'")
    with open(OUTPUT_PATH, "w") as f:
        f.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a unstructured vertical solver using the Dawn compiler")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print the generated SIR",
    )
    main(parser.parse_args())
