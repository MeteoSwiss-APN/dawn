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
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils
from google.protobuf.json_format import MessageToJson, Parse

OUTPUT_NAME = "generate_versioned_field"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"


def main(args: argparse.Namespace):
    interval = sir_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)

    line_1 = sir_utils.make_assignment_stmt(
        sir_utils.make_field_access_expr("a"), sir_utils.make_binary_operator(sir_utils.make_binary_operator(
            sir_utils.make_field_access_expr("b"),
            "/",
            sir_utils.make_field_access_expr("c"),
        ), "+", sir_utils.make_literal_access_expr("5", SIR.BuiltinType.Float)), "=")

    line_2 = sir_utils.make_block_stmt(sir_utils.make_assignment_stmt(
        sir_utils.make_field_access_expr("a"), sir_utils.make_field_access_expr("b"), "="))

    line_3 = sir_utils.make_block_stmt(sir_utils.make_assignment_stmt(
        sir_utils.make_field_access_expr("c"),
        sir_utils.make_binary_operator(
            sir_utils.make_field_access_expr("a"), "+", sir_utils.make_literal_access_expr("1", SIR.BuiltinType.Float)),
        "="))

    body_ast = sir_utils.make_ast(
        [
            line_1,
            sir_utils.make_if_stmt(sir_utils.make_expr_stmt(sir_utils.make_field_access_expr("d")), line_2,
                                   sir_utils.make_block_stmt(sir_utils.make_if_stmt(
                                       sir_utils.make_expr_stmt(
                                           sir_utils.make_field_access_expr("e")),
                                       line_3
                                   ))
                                   )
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        SIR.GridType.Value("Unstructured"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field(
                        "a",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "b",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "c",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "d",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "e",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
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
        description="Generate a stencil that generates a versioned field"
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
