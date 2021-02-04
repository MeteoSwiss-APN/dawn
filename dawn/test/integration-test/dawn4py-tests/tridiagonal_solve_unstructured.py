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
from dawn4py.serialization import SIR, AST
from dawn4py.serialization import utils as serial_utils
from google.protobuf.json_format import MessageToJson, Parse

OUTPUT_NAME = "unstructured_vertical_solver"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"


def create_vertical_region_stmt1():
    """ create a vertical region statement for the stencil
    """

    interval = serial_utils.make_interval(serial_utils.Interval.Start, serial_utils.Interval.Start, 0, 0)

    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_unstructured_field_access_expr("c"),
                serial_utils.make_binary_operator(
                    serial_utils.make_unstructured_field_access_expr("c"),
                    "/",
                    serial_utils.make_unstructured_field_access_expr("b"),
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_unstructured_field_access_expr("d"),
                serial_utils.make_binary_operator(
                    serial_utils.make_unstructured_field_access_expr("d"),
                    "/",
                    serial_utils.make_unstructured_field_access_expr("b"),
                ),
                "=",
            ),
        ]
    )

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, AST.VerticalRegion.Forward
    )
    return vertical_region_stmt


def create_vertical_region_stmt2():
    """ create a vertical region statement for the stencil
    """

    interval = serial_utils.make_interval(serial_utils.Interval.Start, serial_utils.Interval.End, 1, 0)

    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_var_decl_stmt(
                serial_utils.make_type(serial_utils.BuiltinType.Float),
                "m",
                0,
                "=",
                serial_utils.make_expr(
                    serial_utils.make_binary_operator(
                        serial_utils.make_literal_access_expr("1.0", serial_utils.BuiltinType.Float),
                        "/",
                        serial_utils.make_binary_operator(
                            serial_utils.make_unstructured_field_access_expr("b"),
                            "-",
                            serial_utils.make_binary_operator(
                                serial_utils.make_unstructured_field_access_expr("a"),
                                "*",
                                serial_utils.make_unstructured_field_access_expr(
                                    "c", serial_utils.make_unstructured_offset(False), -1
                                ),
                            ),
                        ),
                    )
                ),
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_unstructured_field_access_expr("c"),
                serial_utils.make_binary_operator(
                    serial_utils.make_unstructured_field_access_expr("c"),
                    "*",
                    serial_utils.make_var_access_expr("m"),
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_unstructured_field_access_expr("d"),
                serial_utils.make_binary_operator(
                    serial_utils.make_binary_operator(
                        serial_utils.make_unstructured_field_access_expr("d"),
                        "-",
                        serial_utils.make_binary_operator(
                            serial_utils.make_unstructured_field_access_expr("a"),
                            "*",
                            serial_utils.make_unstructured_field_access_expr(
                                "d", serial_utils.make_unstructured_offset(False), -1
                            ),
                        ),
                    ),
                    "*",
                    serial_utils.make_var_access_expr("m"),
                ),
                "=",
            ),
        ]
    )

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, AST.VerticalRegion.Forward
    )
    return vertical_region_stmt


def create_vertical_region_stmt3():
    """ create a vertical region statement for the stencil
    """

    interval = serial_utils.make_interval(serial_utils.Interval.Start, serial_utils.Interval.End, 0, -1)

    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_unstructured_field_access_expr("d"),
                serial_utils.make_binary_operator(
                    serial_utils.make_unstructured_field_access_expr("c"),
                    "*",
                    serial_utils.make_unstructured_field_access_expr(
                        "d", serial_utils.make_unstructured_offset(False), 1
                    ),
                ),
                "-=",
            )
        ]
    )

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, AST.VerticalRegion.Backward
    )
    return vertical_region_stmt


def main(args: argparse.Namespace):
    sir = serial_utils.make_sir(
        OUTPUT_FILE,
        AST.GridType.Value("Unstructured"),
        [
            serial_utils.make_stencil(
                OUTPUT_NAME,
                serial_utils.make_ast(
                    [
                        create_vertical_region_stmt1(),
                        create_vertical_region_stmt2(),
                        create_vertical_region_stmt3(),
                    ]
                ),
                [
                    serial_utils.make_field(
                        "a",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Cell")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "b",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Cell")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "c",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Cell")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "d",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Cell")], 1
                        ),
                    ),
                ],
            )
        ],
    )

    # print the SIR       
    if args.verbose:
        print(MessageToJson(sir))

    # compile
    code = dawn4py.compile(sir, backend=dawn4py.CodeGenBackend.CXXNaiveIco)

    # write to file
    print(f"Writing generated code to '{OUTPUT_PATH}'")
    with open(OUTPUT_PATH, "w") as f:
        f.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a unstructured vertical solver using the Dawn compiler"
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
