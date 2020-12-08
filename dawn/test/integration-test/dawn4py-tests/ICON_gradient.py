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

OUTPUT_NAME = "ICON_gradient"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"
SIR_OUTPUT_FILE = f"{OUTPUT_NAME}.sir"


def main(args: argparse.Namespace):
    interval = sir_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)
    
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_loop_stmt(
                sir_utils.make_assignment_stmt(
                    sir_utils.make_field_access_expr("geofac_grg"), 
                    sir_utils.make_literal_access_expr("2.", SIR.BuiltinType.Double)), 
                    [SIR.LocationType.Value("Cell"), SIR.LocationType.Value("Edge"), SIR.LocationType.Value("Cell")],
                    include_center = True),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("p_grad"),
                sir_utils.make_reduction_over_neighbor_expr(
                    "+",
                    sir_utils.make_binary_operator( 
                        sir_utils.make_unstructured_field_access_expr("geofac_grg"), 
                        "*", 
                        sir_utils.make_unstructured_field_access_expr("p_ccpr", horizontal_offset=sir_utils.make_unstructured_offset(True))),
                    init=sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double),
                    chain=[SIR.LocationType.Value(
                        "Cell"), SIR.LocationType.Value("Edge"), SIR.LocationType.Value(
                        "Cell")],
                    include_center = True,
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
                        "p_grad",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "p_ccpr",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "geofac_grg",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell"), SIR.LocationType.Value("Edge"), SIR.LocationType.Value("Cell")], 1, include_center = True
                        ),                       
                    ),
                ],
            ),
        ],
    )

    # print the SIR
    # if args.verbose:
    f = open(SIR_OUTPUT_FILE, "w")
    f.write(MessageToJson(sir))
    f.close()

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
