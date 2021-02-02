#!/usr/bin/env python

##===-----------------------------------------------------------------------------*- Python -*-===##
# _
# | |
# __| | __ ___      ___ ___
# / _` |/ _` \ \ /\ / / '_  |
# | (_| | (_| |\ V  V /| | | |
# \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
# This file is distributed under the MIT License (MIT).
# See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

"""Copy stencil SIR generator

This program creates the SIR corresponding to a copy stencil exercising the unstrucutred iteration space conecpt using magic numbers, via
the SIR serialization Python API.
"""

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR, AST
from dawn4py.serialization import utils as sir_utils
from google.protobuf.json_format import MessageToJson, Parse

OUTPUT_NAME = "global_index_stencil_unstr"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"

# lb = 1,
# nudging = 2,
# interior = 3,
# halo = 4


def main(args: argparse.Namespace):
    interval = sir_utils.make_interval(
        AST.Interval.Start, AST.Interval.End, 0, 0)

    # out = in_1 on inner cells
    body_ast_1 = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out"),
                sir_utils.make_field_access_expr("in_1"),
                "=",
            )
        ]
    )
    vertical_region_stmt_1 = sir_utils.make_vertical_region_decl_stmt(
        body_ast_1, interval, AST.VerticalRegion.Forward, sir_utils.make_magic_num_interval(
            0, 1, 0, 0)
    )

    # out = out + in_2 on inner cells
    #   should be merge-able to last stage
    body_ast_2 = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out"),
                sir_utils.make_binary_operator(
                    sir_utils.make_field_access_expr("out"),
                    "+",
                    sir_utils.make_field_access_expr("in_2"),
                ),
                "="
            )
        ]
    )
    vertical_region_stmt_2 = sir_utils.make_vertical_region_decl_stmt(
        body_ast_2, interval, AST.VerticalRegion.Forward, sir_utils.make_interval(
            2, 3, 0, 0)
    )

    # out = out + in_3 on lateral boundary cells
    # out = in_1 on inner cells
    body_ast_3 = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out"),
                sir_utils.make_field_access_expr("in_3"),
                "=",
            )
        ]
    )
    vertical_region_stmt_3 = sir_utils.make_vertical_region_decl_stmt(
        body_ast_3, interval, AST.VerticalRegion.Forward, sir_utils.make_interval(
            3, 4, 0, 0)
    )

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        AST.GridType.Value("Unstructured"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast(
                    [vertical_region_stmt_1, vertical_region_stmt_2, vertical_region_stmt_3]),
                [
                    sir_utils.make_field(
                        "in_1",
                        sir_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "in_2",
                        sir_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "in_3",
                        sir_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "out",
                        sir_utils.make_field_dimensions_unstructured(
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
    pass_groups = dawn4py.default_pass_groups()
    pass_groups.insert(1, dawn4py.PassGroup.MultiStageMerger)
    pass_groups.insert(1, dawn4py.PassGroup.StageMerger)
    # code = dawn4py.compile(sir, groups=pass_groups,
    #                        backend=dawn4py.CodeGenBackend.CXXNaiveIco, merge_stages=True, merge_do_methods=True)
    code = dawn4py.compile(sir, groups=pass_groups,
                           backend=dawn4py.CodeGenBackend.CXXNaiveIco)

    # write to file
    print(f"Writing generated code to '{OUTPUT_PATH}'")
    with open(OUTPUT_PATH, "w") as f:
        f.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a simple copy-shift stencil using Dawn compiler"
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
