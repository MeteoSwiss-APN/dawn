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

"""
this integration test generates a SIR which will contain an empty stage after the PassRemoveScalars
has run, and thus ensures that such emtpied stages are handled correctly
"""

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR, AST
from dawn4py.serialization import utils as serial_utils
from google.protobuf.json_format import MessageToJson, Parse

OUTPUT_NAME = "empty_stage_stencil"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"


def main(args: argparse.Namespace):
    interval = serial_utils.make_interval(
        AST.Interval.Start, AST.Interval.End, 0, 0)

    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("x"),
                serial_utils.make_literal_access_expr(
                    "1",  AST.BuiltinType.Double),
                "=",
            )
        ]
    )

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, AST.VerticalRegion.Forward
    )

    sir = serial_utils.make_sir(
        OUTPUT_FILE,
        AST.GridType.Value("Unstructured"),
        [
            serial_utils.make_stencil(
                OUTPUT_NAME,
                serial_utils.make_ast([vertical_region_stmt]),
                [
                    serial_utils.make_field(
                        "in",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "out",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "x",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                        True
                    ),
                ],
            ),
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
        description="Generate a stencil that degenerates to an empty stage"
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
