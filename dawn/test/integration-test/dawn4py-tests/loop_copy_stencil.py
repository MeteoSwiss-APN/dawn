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

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR, AST
from dawn4py.serialization import utils as serial_utils
from google.protobuf.json_format import MessageToJson

OUTPUT_NAME = "loop_copy_stencil"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"


def main(args: argparse.Namespace):
    interval = serial_utils.make_interval(
        AST.Interval.Start, AST.Interval.End, 0, 0)

    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_loop_stmt(
                [serial_utils.make_assignment_stmt(
                    serial_utils.make_field_access_expr("out"),
                    serial_utils.make_field_access_expr("in"),
                    "=",
                )],
                [AST.LocationType.Value("Cell"),
                 AST.LocationType.Value("Edge")]
            )
        ]
    )

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, AST.VerticalRegion.Forward)

    sir = serial_utils.make_sir(
        OUTPUT_FILE,
        AST.GridType.Value("Unstructured"),
        [
            serial_utils.make_stencil(
                OUTPUT_NAME,
                serial_utils.make_ast([vertical_region_stmt]),
                [
                    serial_utils.make_field(
                        "out",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Cell"),
                             AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "in",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
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
        description="Generate a simple copy stencil using Dawn compiler")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print the generated SIR",
    )
    main(parser.parse_args())
