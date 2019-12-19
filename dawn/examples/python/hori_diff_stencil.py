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

"""Horizontal diffusion stencil SIR generator

This program creates the SIR corresponding to an horizontal diffusion stencil using the SIR serialization Python API.
The horizontal diffusion is a basic example that contains horizontal data dependencies that need to be resolved
by the compiler passes.
The code is meant as an example for high-level DSLs that could generate SIR from their own
internal IR.
The program contains two parts:
    1. construct the SIR of the example
    2. pass the SIR to the dawn compiler in order to run all optimizer passes and code generation.
       In this example the compiler is configured with the CUDA backend, therefore will code
       generate an optimized CUDA implementation.

"""

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils

OUTPUT_NAME = "hori_diff_stencil"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", f"{OUTPUT_NAME}.cpp")


def main(args: argparse.Namespace):
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

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        SIR.GridType.Value("Cartesian"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field("in"),
                    sir_utils.make_field("out"),
                    sir_utils.make_field("coeff"),
                    sir_utils.make_field("lap", is_temporary=True),
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
    parser = argparse.ArgumentParser(
        description="Generate a simple horizontal diffusion stencil using Dawn compiler"
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
