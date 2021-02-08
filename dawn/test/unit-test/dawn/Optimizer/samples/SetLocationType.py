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

"""Generate input for SetLocationType tests"""

import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as serial_utils
from google.protobuf.json_format import MessageToJson, Parse


def copy_fields():
    outputfile = "../input/test_set_stage_location_type_copy_fields.sir"
    interval = serial_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)

    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("out_cell"),
                serial_utils.make_field_access_expr("in_cell"),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("out_edge"),
                serial_utils.make_field_access_expr("in_edge"),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("out_vertex"),
                serial_utils.make_field_access_expr("in_vertex"),
                "=",
            )
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
                    serial_utils.make_field(
                        "in_vertex",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "out_vertex",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Vertex")], 1
                        ),
                    ),
                ],
            ),
        ],
    )
    f = open(outputfile, "w")
    f.write(MessageToJson(sir))
    f.close()


def copy_vars():
    outputfile = "../input/test_set_stage_location_type_copy_vars.sir"

    interval = serial_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)

    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_var_decl_stmt(
                serial_utils.make_type(serial_utils.BuiltinType.Float),
                "out_var_cell"),
            serial_utils.make_var_decl_stmt(
                serial_utils.make_type(serial_utils.BuiltinType.Float),
                "out_var_edge"),
            serial_utils.make_var_decl_stmt(
                serial_utils.make_type(serial_utils.BuiltinType.Float),
                "out_var_vertex"),
            serial_utils.make_assignment_stmt(
                serial_utils.make_var_access_expr("out_var_cell"),
                serial_utils.make_field_access_expr("in_cell"),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_var_access_expr("out_var_edge"),
                serial_utils.make_field_access_expr("in_edge"),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_var_access_expr("out_var_vertex"),
                serial_utils.make_field_access_expr("in_vertex"),
                "=",
            )
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
                        "in_edge",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "in_vertex",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Vertex")], 1
                        ),
                    ),
                ],
            ),
        ],
    )

    f = open(outputfile, "w")
    f.write(MessageToJson(sir))
    f.close()


def if_stmt():
    outputfile = "../input/test_set_stage_location_type_if_stmt.sir"

    interval = serial_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)

    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_var_decl_stmt(
                serial_utils.make_type(serial_utils.BuiltinType.Float),
                "out_var_cell"),
            serial_utils.make_if_stmt(serial_utils.make_expr_stmt(serial_utils.make_var_access_expr("out_var_cell")), serial_utils.make_block_stmt(serial_utils.make_assignment_stmt(
                serial_utils.make_var_access_expr("out_var_cell"),
                serial_utils.make_field_access_expr("in_cell"),
                "=",
            ))),
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
                ],
            ),
        ],
    )

    f = open(outputfile, "w")
    f.write(MessageToJson(sir))
    f.close()


def function_call():
    outputfile = "../input/test_set_stage_location_type_function_call.sir"

    interval = serial_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)

    fun_ast = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("out"),
                serial_utils.make_literal_access_expr(
                    value="2.0", type=serial_utils.BuiltinType.Float),
                "=",
            ),
        ]
    )

    arg_field = serial_utils.make_field(
        "out",
        serial_utils.make_field_dimensions_unstructured(
            [SIR.LocationType.Value("Cell")], 1
        )
    )

    fun = serial_utils.make_stencil_function(
        name='f', asts=[fun_ast], intervals=[interval], arguments=[serial_utils.make_stencil_function_arg(arg_field)])

    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_expr_stmt(expr=serial_utils.make_stencil_fun_call_expr(
                callee="f", arguments=[serial_utils.make_field_access_expr("out_cell")])),
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
                        "out_cell",
                        serial_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                ],
            ),
        ],
        functions=[fun]
    )

    f = open(outputfile, "w")
    f.write(MessageToJson(sir))
    f.close()


if __name__ == "__main__":
    copy_fields()
    copy_vars()
    if_stmt()
    function_call()
