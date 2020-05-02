#!/usr/bin/env python3

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

"""Simple reduction stencil SIR generator

This program creates the SIR corresponding to a simple unstructured reduction stencil using the SIR serialization Python API.

"""

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils

STENCIL_NAME = "simple_reduction_stencil"
OUTPUT_FILE = f"{STENCIL_NAME}.sir"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the SIR of a simple reduction stencil")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print the generated SIR",
    )

    args = parser.parse_args()

    interval = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)

    # create the out = reduce(+, init=1.0, rhs=in, E->C) statement
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out"),
                sir_utils.make_reduction_over_neighbor_expr(
                    "+",
                    rhs=sir_utils.make_field_access_expr("in"),
                    init=sir_utils.make_literal_access_expr("1.0", SIR.BuiltinType.Float),
                    chain=[SIR.LocationType.Value("Edge"), SIR.LocationType.Value("Cell")],
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
                STENCIL_NAME,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field(
                        "in",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "out",
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
        sir_utils.pprint(sir)

    with open(OUTPUT_FILE, mode="w") as f:
        f.write(sir_utils.to_json(sir))
