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
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils
from google.protobuf.json_format import MessageToJson, Parse

OUTPUT_NAME = "offset_reduction"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"


def main(args: argparse.Namespace):
    interval = sir_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)

    # create the out = in[i+1] statement
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_unstructured_field_access_expr("out_vn_e"),
                sir_utils.make_reduction_over_neighbor_expr(
                    "+",
                    sir_utils.make_binary_operator(
                    sir_utils.make_unstructured_field_access_expr(
                        "raw_diam_coeff", horizontal_offset=sir_utils.make_unstructured_offset(True)),
                    "*",
                    sir_utils.make_unstructured_field_access_expr(
                        "prism_thick_e", horizontal_offset=sir_utils.make_unstructured_offset(True)),
                    ),
                    sir_utils.make_literal_access_expr(
                        ".0", SIR.BuiltinType.Float),                    
                    chain=[SIR.LocationType.Value("Edge"), SIR.LocationType.Value("Cell"), SIR.LocationType.Value("Edge")],
                    weights=[sir_utils.make_unstructured_field_access_expr(
                        "e2c_aux", horizontal_offset=sir_utils.make_unstructured_offset(True)),
                        sir_utils.make_unstructured_field_access_expr(
                        "e2c_aux", horizontal_offset=sir_utils.make_unstructured_offset(True)),
                        sir_utils.make_unstructured_field_access_expr(
                        "e2c_aux", horizontal_offset=sir_utils.make_unstructured_offset(True)),
                        sir_utils.make_unstructured_field_access_expr(
                        "e2c_aux", horizontal_offset=sir_utils.make_unstructured_offset(True))],
                    offsets=[0, 0, 1, 1]
                ),
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
                        "out_vn_e",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "raw_diam_coeff",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge"), 
                             SIR.LocationType.Value("Cell"), 
                             SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "prism_thick_e",
                        sir_utils.make_field_dimensions_unstructured(                            
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "e2c_aux",
                        sir_utils.make_field_dimensions_unstructured(                            
                            [SIR.LocationType.Value("Edge"), SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                ],
            ),
        ],
    )

    # print the SIR
    f = open("offset_reduction.sir", "w")
    f.write(MessageToJson(sir))
    f.close()
   
    # compile
    code = dawn4py.compile(sir, backend=dawn4py.CodeGenBackend.CXXNaiveIco)

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
