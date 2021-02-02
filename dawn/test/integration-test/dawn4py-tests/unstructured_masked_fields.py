#!/usr/bin/env python

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

This program creates the HIR corresponding to an unstructured stencil using the SIR serialization Python API.
The code is meant as an example for high-level DSLs that could generate HIR from their own
internal IR.
"""

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR, AST
from dawn4py.serialization import utils as sir_utils
from google.protobuf.json_format import MessageToJson, Parse

OUTPUT_NAME = "unstructured_masked_fields"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"


def main(args: argparse.Namespace):
    interval = sir_utils.make_interval(
        AST.Interval.Start, AST.Interval.End, 0, 0)

    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out"),
                sir_utils.make_binary_operator(
                    sir_utils.make_field_access_expr("full"),
                    "+",
                    sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("horizontal"),
                        "+",
                        sir_utils.make_field_access_expr("vertical"))),
                "="),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out"),
                sir_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=sir_utils.make_literal_access_expr(
                        "1.0", AST.BuiltinType.Float),
                    rhs=sir_utils.make_field_access_expr(
                        "horizontal_sparse", [True, 0]),
                    chain=[AST.LocationType.Value(
                        "Edge"), AST.LocationType.Value("Cell")],
                ),
                "=",
            )
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, AST.VerticalRegion.Forward
    )

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        AST.GridType.Value("Unstructured"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field(
                        "out",
                        sir_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "full",
                        sir_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "horizontal",
                        sir_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 0
                        ),
                    ),
                    sir_utils.make_field(
                        "horizontal_sparse",
                        sir_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value(
                                "Edge"), AST.LocationType.Value("Cell")], 0
                        ),
                    ),
                    sir_utils.make_vertical_field("vertical"),
                ],
            ),
        ],
    )

    # print the SIR       
    if args.verbose:
        print(MessageToJson(sir))

    with open("out.json", "w+") as f:
        f.write(sir_utils.to_json(sir))

    # compile
    code = dawn4py.compile(sir, backend=dawn4py.CodeGenBackend.CUDAIco)

    # write to file
    print(f"Writing generated code to '{OUTPUT_PATH}'")
    with open(OUTPUT_PATH, "w") as f:
        f.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a simple unstructured copy stencil using Dawn compiler"
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
