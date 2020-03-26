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

import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils
from google.protobuf.json_format import MessageToJson

stencil_name = "sparse_dimensions"
output_file = f"{stencil_name}.cpp"


if __name__ == "__main__":
    interval = sir_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)

    # create the out = reduce(sparse_CE * in) statement
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out"),
                sir_utils.make_reduction_over_neighbor_expr(
                    "+",
                    rhs=sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("sparse_CE"), "*", sir_utils.make_field_access_expr("in")),
                    init=sir_utils.make_literal_access_expr(
                        "1.0", SIR.BuiltinType.Float),
                    chain=[SIR.LocationType.Value('Cell'),SIR.LocationType.Value('Edge')]
                ),
                "=",
            )
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward)

    sir = sir_utils.make_sir(
        output_file,
        SIR.GridType.Value("Unstructured"),
        [
            sir_utils.make_stencil(
                stencil_name,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field("in", sir_utils.make_field_dimensions_unstructured(
                        [SIR.LocationType.Value('Edge')], 1)),
                    sir_utils.make_field("sparse_CE", sir_utils.make_field_dimensions_unstructured(
                        [SIR.LocationType.Value('Cell'), SIR.LocationType.Value('Edge')], 1)),
                    sir_utils.make_field("out", sir_utils.make_field_dimensions_unstructured(
                        [SIR.LocationType.Value('Cell')], 1))
                ],
            ),
        ],
    )

    dawn4py.compile(sir, backend="c++-naive", serialize_iir=True, output_file=output_file)
    os.rename(stencil_name + ".0.iir", stencil_name + ".iir")
