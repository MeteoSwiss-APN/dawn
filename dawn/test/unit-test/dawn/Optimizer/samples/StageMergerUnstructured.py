# -*- coding: utf-8 -*-
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

"""Generate input for StagerMerger tests"""

import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as serial_utils
from google.protobuf.json_format import MessageToJson, Parse

backend = "c++-naive-ico"


def two_copies():
    outputfile = "StageMergerTestTwoCopies"
    interval = serial_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)

    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_unstructured_field_access_expr("out_cell_1"),
                serial_utils.make_unstructured_field_access_expr("in_cell_1"),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_unstructured_field_access_expr("out_cell_2"),
                serial_utils.make_unstructured_field_access_expr("in_cell_2"),
                "=",
            ),
        ]
    )

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )

    sir = serial_utils.make_sir(
        outputfile,
        SIR.GridType.Value("Unstructured"),
        [
            serial_utils.make_stencil(
                "generated",
                serial_utils.make_ast([vertical_region_stmt]),
                [
                    serial_utils.make_field(
                        "in_cell_1",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "out_cell_1",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "in_cell_2",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "out_cell_2",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                ],
            ),
        ],
    )
    sim = dawn4py.lower_and_optimize(sir, dawn4py.default_pass_groups())
    with open(outputfile, mode="w") as f:
        f.write(MessageToJson(sim["generated"]))
    os.rename(outputfile, "../input/" + outputfile + ".iir")


def two_copies_mixed():
    outputfile = "StageMergerTestTwoCopiesMixed"
    interval = serial_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)

    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_unstructured_field_access_expr("out_cell"),
                serial_utils.make_unstructured_field_access_expr("in_cell"),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_unstructured_field_access_expr("out_edge"),
                serial_utils.make_unstructured_field_access_expr("in_edge"),
                "=",
            ),
        ]
    )

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )

    sir = serial_utils.make_sir(
        outputfile,
        SIR.GridType.Value("Unstructured"),
        [
            serial_utils.make_stencil(
                "generated",
                serial_utils.make_ast([vertical_region_stmt]),
                [
                    serial_utils.make_field(
                        "in_cell",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "out_cell",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "in_edge",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "out_edge",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                ],
            ),
        ],
    )
    sim = dawn4py.lower_and_optimize(sir, dawn4py.default_pass_groups())
    with open(outputfile, mode="w") as f:
        f.write(MessageToJson(sim["generated"]))
    os.rename(outputfile, "../input/" + outputfile + ".iir")


if __name__ == "__main__":
    two_copies()
    two_copies_mixed()
