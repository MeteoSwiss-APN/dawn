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
"""Tridiagonal solve computation SIR generator

This program creates the SIR corresponding to a tridiagonal solve computation using the Python API.
The tridiagonal solve is a basic example that contains vertical data dependencies that need to be resolved
by the compiler passes.
"""

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils
from google.protobuf.json_format import MessageToJson


stencil_name = "tridiagonal_solve_unstructured"
output_file = f"{stencil_name}.sir"


def create_vertical_region_stmt1():
    """ create a vertical region statement for the stencil
    """

    interval = sir_utils.make_interval(sir_utils.Interval.Start, sir_utils.Interval.Start, 0, 0)

    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_unstructured_field_access_expr("c"),
                sir_utils.make_binary_operator(
                    sir_utils.make_unstructured_field_access_expr("c"),
                    "/",
                    sir_utils.make_unstructured_field_access_expr("b"),
                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_unstructured_field_access_expr("d"),
                sir_utils.make_binary_operator(
                    sir_utils.make_unstructured_field_access_expr("d"),
                    "/",
                    sir_utils.make_unstructured_field_access_expr("b"),
                ),
                "=",
            ),
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )
    return vertical_region_stmt


def create_vertical_region_stmt2():
    """ create a vertical region statement for the stencil
    """

    interval = sir_utils.make_interval(sir_utils.Interval.Start, sir_utils.Interval.End, 1, 0)

    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_var_decl_stmt(
                sir_utils.make_type(sir_utils.BuiltinType.Float),
                "m",
                0,
                "=",
                sir_utils.make_expr(
                    sir_utils.make_binary_operator(
                        sir_utils.make_literal_access_expr("1.0", sir_utils.BuiltinType.Float),
                        "/",
                        sir_utils.make_binary_operator(
                            sir_utils.make_unstructured_field_access_expr("b"),
                            "-",
                            sir_utils.make_binary_operator(
                                sir_utils.make_unstructured_field_access_expr("a"),
                                "*",
                                sir_utils.make_unstructured_field_access_expr(
                                    "c", sir_utils.make_unstructured_offset(False), -1
                                ),
                            ),
                        ),
                    )
                ),
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_unstructured_field_access_expr("c"),
                sir_utils.make_binary_operator(
                    sir_utils.make_unstructured_field_access_expr("c"),
                    "*",
                    sir_utils.make_var_access_expr("m"),
                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_unstructured_field_access_expr("d"),
                sir_utils.make_binary_operator(
                    sir_utils.make_binary_operator(
                        sir_utils.make_unstructured_field_access_expr("d"),
                        "-",
                        sir_utils.make_binary_operator(
                            sir_utils.make_unstructured_field_access_expr("a"),
                            "*",
                            sir_utils.make_unstructured_field_access_expr(
                                "d", sir_utils.make_unstructured_offset(False), -1
                            ),
                        ),
                    ),
                    "*",
                    sir_utils.make_var_access_expr("m"),
                ),
                "=",
            ),
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )
    return vertical_region_stmt


def create_vertical_region_stmt3():
    """ create a vertical region statement for the stencil
    """

    interval = sir_utils.make_interval(sir_utils.Interval.Start, sir_utils.Interval.End, 0, -1)

    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_unstructured_field_access_expr("d"),
                sir_utils.make_binary_operator(
                    sir_utils.make_unstructured_field_access_expr("c"),
                    "*",
                    sir_utils.make_unstructured_field_access_expr(
                        "d", sir_utils.make_unstructured_offset(False), 1
                    ),
                ),
                "-=",
            )
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Backward
    )
    return vertical_region_stmt


def main(args: argparse.Namespace):
    sir = sir_utils.make_sir(
        output_file,
        SIR.GridType.Value("Unstructured"),
        [
            sir_utils.make_stencil(
                stencil_name,
                sir_utils.make_ast(
                    [
                        create_vertical_region_stmt1(),
                        create_vertical_region_stmt2(),
                        create_vertical_region_stmt3(),
                    ]
                ),
                [
                    sir_utils.make_field(
                        "a",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "b",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "c",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "d",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                ],
            )
        ],
    )

    # print the SIR
    if args.verbose:
        sir_utils.pprint(sir)

    f = open(output_file, "w")
    f.write(MessageToJson(sir))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the SIR of an unstructured vertical solver"
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
