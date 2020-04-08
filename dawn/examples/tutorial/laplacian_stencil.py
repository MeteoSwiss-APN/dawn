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

"""Copy stencil HIR generator

This program creates the HIR corresponding to a copy stencil using the Python API of the HIR.
The copy stencil is a hello world for stencil computations.
The code is meant as an example for high-level DSLs that could generate HIR from their own
internal IR.
The program contains two parts:
    1. construct the HIR of the example
    2. pass the HIR to the dawn compiler in order to run all optimizer passes and code generation.
       In this example the compiler is configured with the CUDA backend, therefore will code
       generate an optimized CUDA implementation.

"""


import argparse
import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils

OUTPUT_NAME = "laplacian_stencil"
OUTPUT_FILE = f"{OUTPUT_NAME}_from_python.cpp"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)


def main(args: argparse.Namespace):
    interval = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)

    # create the laplace statement
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out", [0, 0, 0]),
                sir_utils.make_binary_operator(
                    sir_utils.make_binary_operator(
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr("in", [0, 0, 0]),
                            "*",
                            sir_utils.make_literal_access_expr(
                                "-4.0", sir_utils.BuiltinType.Float
                            ),
                        ),
                        "+",
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
                    "/",
                    sir_utils.make_binary_operator(
                        sir_utils.make_var_access_expr("dx", is_external=True),
                        "*",
                        sir_utils.make_var_access_expr("dx", is_external=True),
                    ),
                ),
                "=",
            ),
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )

    stencils_globals = sir_utils.GlobalVariableMap()
    stencils_globals.map["dx"].double_value = 0.0

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        SIR.GridType.Value("Cartesian"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field("out", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("in", sir_utils.make_field_dimensions_cartesian()),
                ],
            )
        ],
        global_variables=stencils_globals,
    )

    # print the SIR
    if args.verbose:
        sir_utils.pprint(sir)

    # serialize the SIR to file
    sir_file = open("./laplacian_stencil_from_python.sir", "wb")
    sir_file.write(sir_utils.to_bytes(sir))
    sir_file.close()

    # compile
    code = dawn4py.compile_sir(
        sir_utils.to_bytes(sir), codegen_backend=dawn4py.CodeGenBackend.CXXNaive
    )

    # write to file
    print(f"Writing generated code to '{OUTPUT_PATH}'")
    with open(OUTPUT_PATH, "w") as f:
        f.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a simple laplace stencil using Dawn compiler"
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
