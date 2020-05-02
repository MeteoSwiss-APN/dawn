#!/usr/bin/env python3

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

"""Copy stencil SIR generator

This program creates the SIR corresponding to a copy-shift stencil using the SIR serialization Python API.
"""

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils

STENCIL_NAME = "copy_stencil"
OUTPUT_FILE = f"{STENCIL_NAME}.sir"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the SIR of a simple copy-shift stencil")
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

    # create the out = in[i+1] statement
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out", [0, 0, 0]),
                sir_utils.make_field_access_expr("in", [1, 0, 0]),
                "=",
            )
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        SIR.GridType.Value("Cartesian"),
        [
            sir_utils.make_stencil(
                STENCIL_NAME,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field("in", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("out", sir_utils.make_field_dimensions_cartesian()),
                ],
            )
        ],
    )

    # print the SIR
    if args.verbose:
        sir_utils.pprint(sir)

    with open(OUTPUT_FILE, mode="w") as f:
        f.write(sir_utils.to_json(sir))
