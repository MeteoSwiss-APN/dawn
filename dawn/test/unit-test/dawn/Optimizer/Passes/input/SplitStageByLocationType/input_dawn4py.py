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

This program creates the SIR corresponding to a copy stencil using the SIR serialization Python API.
The copy stencil is a 'hello world' for stencil computations.
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
from google.protobuf.json_format import MessageToJson, Parse

OUTPUT_NAME = "copy_stencil"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"


def main(args: argparse.Namespace):
    interval = sir_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)

    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out_cell"),
                sir_utils.make_field_access_expr("in_cell"),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out_edge"),
                sir_utils.make_field_access_expr("in_edge"),
                "=",
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
                        "in_cell",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "out_cell",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "in_edge",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "out_edge",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                ],
            ),
        ],
    )

    # output SIR to file
    f = open("split_stage_by_location_type_test_stencil_01.sir", "w")
    f.write(MessageToJson(sir))
    f.close()

    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_var_decl_stmt(
                sir_utils.make_type(sir_utils.BuiltinType.Float),
                "out_var_cell"),
            sir_utils.make_var_decl_stmt(
                sir_utils.make_type(sir_utils.BuiltinType.Float),
                "out_var_edge"),
            sir_utils.make_assignment_stmt(
                sir_utils.make_var_access_expr("out_var_cell"),
                sir_utils.make_field_access_expr("in_cell"),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_var_access_expr("out_var_edge"),
                sir_utils.make_field_access_expr("in_edge"),
                "=",
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
                        "in_cell",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "in_edge",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                ],
            ),
        ],
    )

    # output SIR to file
    f = open("split_stage_by_location_type_test_stencil_02.sir", "w")
    f.write(MessageToJson(sir))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a simple copy-shift stencil using Dawn compiler")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print the generated SIR",
    )
    main(parser.parse_args())
