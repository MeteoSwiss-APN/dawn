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
from dawn4py.serialization import SIR, AST
from dawn4py.serialization import utils as serial_utils
from google.protobuf.json_format import MessageToJson, Parse

OUTPUT_NAME = "hori_diff_stencil"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"


def main(args: argparse.Namespace):
    interval = serial_utils.make_interval(AST.Interval.Start, AST.Interval.End, 0, 0)

    # create the stencil body AST
    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("lap"),
                serial_utils.make_binary_operator(
                    serial_utils.make_binary_operator(
                        serial_utils.make_literal_access_expr("-4.0", AST.BuiltinType.Float),
                        "*",
                        serial_utils.make_field_access_expr("in"),
                    ),
                    "+",
                    serial_utils.make_binary_operator(
                        serial_utils.make_field_access_expr("coeff"),
                        "*",
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr("in", [1, 0, 0]),
                            "+",
                            serial_utils.make_binary_operator(
                                serial_utils.make_field_access_expr("in", [-1, 0, 0]),
                                "+",
                                serial_utils.make_binary_operator(
                                    serial_utils.make_field_access_expr("in", [0, 1, 0]),
                                    "+",
                                    serial_utils.make_field_access_expr("in", [0, -1, 0]),
                                ),
                            ),
                        ),
                    ),
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("out"),
                serial_utils.make_binary_operator(
                    serial_utils.make_binary_operator(
                        serial_utils.make_literal_access_expr("-4.0", AST.BuiltinType.Float),
                        "*",
                        serial_utils.make_field_access_expr("lap"),
                    ),
                    "+",
                    serial_utils.make_binary_operator(
                        serial_utils.make_field_access_expr("coeff"),
                        "*",
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr("lap", [1, 0, 0]),
                            "+",
                            serial_utils.make_binary_operator(
                                serial_utils.make_field_access_expr("lap", [-1, 0, 0]),
                                "+",
                                serial_utils.make_binary_operator(
                                    serial_utils.make_field_access_expr("lap", [0, 1, 0]),
                                    "+",
                                    serial_utils.make_field_access_expr("lap", [0, -1, 0]),
                                ),
                            ),
                        ),
                    ),
                ),
                "=",
            ),
        ]
    )

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, AST.VerticalRegion.Forward
    )

    sir = serial_utils.make_sir(
        OUTPUT_FILE,
        AST.GridType.Value("Cartesian"),
        [
            serial_utils.make_stencil(
                OUTPUT_NAME,
                serial_utils.make_ast([vertical_region_stmt]),
                [
                    serial_utils.make_field("in", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field("out", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field("coeff", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field(
                        "lap", serial_utils.make_field_dimensions_cartesian(), is_temporary=True
                    ),
                ],
            )
        ],
    )

    # print the SIR       
    if args.verbose:
        print(MessageToJson(sir))

    # compile
    code = dawn4py.compile(sir, backend=dawn4py.CodeGenBackend.CUDA)

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
