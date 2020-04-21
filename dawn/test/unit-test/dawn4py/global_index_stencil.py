#!/usr/bin/python3
# -*- coding: utf-8 -*-
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

"""Global indexing SIR generator

This program creates the SIR corresponding to an example usage of global indexing
using the SIR serialization Python API.
"""

import argparse
import os.path

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils
from google.protobuf.json_format import MessageToJson

stencil_name = "global_index_stencil"
output_file = f"{stencil_name}.sir"


def create_vertical_region_stmt() -> SIR.VerticalRegionDeclStmt:
    """ create a vertical region statement for the stencil
    """

    interval = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)

    # create the out = in statement
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out", [0, 0, 0]),
                sir_utils.make_field_access_expr("in", [0, 0, 0]),
                "=",
            )
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )
    return vertical_region_stmt


def create_boundary_correction_region(
    value="0", i_interval=None, j_interval=None
) -> SIR.VerticalRegionDeclStmt:
    interval = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)
    boundary_body = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out", [0, 0, 0]),
                sir_utils.make_literal_access_expr(value, SIR.BuiltinType.Float),
                "=",
            )
        ]
    )
    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        boundary_body, interval, SIR.VerticalRegion.Forward, IRange=i_interval, JRange=j_interval
    )
    return vertical_region_stmt


def main(args: argparse.Namespace):
    sir = sir_utils.make_sir(
        output_file,
        SIR.GridType.Value("Cartesian"),
        [
            sir_utils.make_stencil(
                stencil_name,
                sir_utils.make_ast(
                    [
                        create_vertical_region_stmt(),
                        create_boundary_correction_region(
                            value="4",
                            i_interval=sir_utils.make_interval(
                                SIR.Interval.End, SIR.Interval.End, -1, 0
                            ),
                        ),
                        create_boundary_correction_region(
                            value="8",
                            i_interval=sir_utils.make_interval(
                                SIR.Interval.Start, SIR.Interval.Start, 0, 1
                            ),
                        ),
                        create_boundary_correction_region(
                            value="6",
                            j_interval=sir_utils.make_interval(
                                SIR.Interval.End, SIR.Interval.End, -1, 0
                            ),
                        ),
                        create_boundary_correction_region(
                            value="2",
                            j_interval=sir_utils.make_interval(
                                SIR.Interval.Start, SIR.Interval.Start, 0, 1
                            ),
                        ),
                        create_boundary_correction_region(
                            value="1",
                            j_interval=sir_utils.make_interval(
                                SIR.Interval.Start, SIR.Interval.Start, 0, 1
                            ),
                            i_interval=sir_utils.make_interval(
                                SIR.Interval.Start, SIR.Interval.Start, 0, 1
                            ),
                        ),
                        create_boundary_correction_region(
                            value="3",
                            j_interval=sir_utils.make_interval(
                                SIR.Interval.Start, SIR.Interval.Start, 0, 1
                            ),
                            i_interval=sir_utils.make_interval(
                                SIR.Interval.End, SIR.Interval.End, -1, 0
                            ),
                        ),
                        create_boundary_correction_region(
                            value="7",
                            j_interval=sir_utils.make_interval(
                                SIR.Interval.End, SIR.Interval.End, -1, 0
                            ),
                            i_interval=sir_utils.make_interval(
                                SIR.Interval.Start, SIR.Interval.Start, 0, 1
                            ),
                        ),
                        create_boundary_correction_region(
                            value="5",
                            j_interval=sir_utils.make_interval(
                                SIR.Interval.End, SIR.Interval.End, -1, 0
                            ),
                            i_interval=sir_utils.make_interval(
                                SIR.Interval.End, SIR.Interval.End, -1, 0
                            ),
                        ),
                    ]
                ),
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

    with open(output_file, mode="w") as f:
        f.write(sir_utils.to_json(sir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the SIR of a simple stencil that uses global indexing"
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
