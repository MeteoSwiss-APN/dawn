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


"""pytest definitions and utilities"""

from dawn4py.serialization import SIR, AST
from dawn4py.serialization import utils as serial_utils


GRID_TEST_CASES = ["copy_stencil",
                   "hori_diff_stencil", "tridiagonal_solve_stencil"]

UNSTRUCTURED_TEST_CASES = ["unstructured_stencil"]


# ---- Grid cases -----
def make_copy_stencil_sir(name=None):
    OUTPUT_NAME = name if name is not None else "copy_stencil"
    OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"

    interval = serial_utils.make_interval(
        AST.Interval.Start, AST.Interval.End, 0, 0)

    # create the out = in[i+1] statement
    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("out", [0, 0, 0]),
                serial_utils.make_field_access_expr("in", [1, 0, 0]),
                "=",
            )
        ]
    )

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, AST.VerticalRegion.Forward)

    sir = serial_utils.make_sir(
        OUTPUT_FILE,
        serial_utils.GridType.Value("Cartesian"),
        [
            serial_utils.make_stencil(
                OUTPUT_NAME,
                serial_utils.make_ast([vertical_region_stmt]),
                [
                    serial_utils.make_field(
                        "in", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field("out", serial_utils.make_field_dimensions_cartesian())],
            )
        ],
    )

    return sir


def make_hori_diff_stencil_sir(name=None):
    OUTPUT_NAME = name if name is not None else "hori_diff_stencil"
    OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"

    interval = serial_utils.make_interval(
        AST.Interval.Start, AST.Interval.End, 0, 0)

    # create the stencil body AST
    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("lap"),
                serial_utils.make_binary_operator(
                    serial_utils.make_binary_operator(
                        serial_utils.make_literal_access_expr(
                            "-4.0", AST.BuiltinType.Float),
                        "*",
                        serial_utils.make_field_access_expr("in"),
                    ),
                    "+",
                    serial_utils.make_binary_operator(
                        serial_utils.make_field_access_expr("coeff"),
                        "*",
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr("in", [1, 0, 0]),
                            "+",
                            serial_utils.make_binary_operator(
                                serial_utils.make_field_access_expr(
                                    "in", [-1, 0, 0]),
                                "+",
                                serial_utils.make_binary_operator(
                                    serial_utils.make_field_access_expr(
                                        "in", [0, 1, 0]),
                                    "+",
                                    serial_utils.make_field_access_expr(
                                        "in", [0, -1, 0]),
                                ),
                            ),
                        ),
                    ),
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("out"),
                serial_utils.make_binary_operator(
                    serial_utils.make_binary_operator(
                        serial_utils.make_literal_access_expr(
                            "-4.0", AST.BuiltinType.Float),
                        "*",
                        serial_utils.make_field_access_expr("lap"),
                    ),
                    "+",
                    serial_utils.make_binary_operator(
                        serial_utils.make_field_access_expr("coeff"),
                        "*",
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr("lap", [1, 0, 0]),
                            "+",
                            serial_utils.make_binary_operator(
                                serial_utils.make_field_access_expr(
                                    "lap", [-1, 0, 0]),
                                "+",
                                serial_utils.make_binary_operator(
                                    serial_utils.make_field_access_expr(
                                        "lap", [0, 1, 0]),
                                    "+",
                                    serial_utils.make_field_access_expr(
                                        "lap", [0, -1, 0]),
                                ),
                            ),
                        ),
                    ),
                ),
                "=",
            ),
        ]
    )

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, AST.VerticalRegion.Forward)

    sir = serial_utils.make_sir(
        OUTPUT_FILE,
        serial_utils.GridType.Value("Cartesian"),
        [
            serial_utils.make_stencil(
                OUTPUT_NAME,
                serial_utils.make_ast([vertical_region_stmt]),
                [
                    serial_utils.make_field(
                        "in", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field(
                        "out", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field(
                        "coeff", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field(
                        "lap", serial_utils.make_field_dimensions_cartesian(), is_temporary=True),
                ],
            )
        ],
    )

    return sir


def make_tridiagonal_solve_stencil_sir(name=None):
    OUTPUT_NAME = name if name is not None else "tridiagonal_solve_stencil"
    OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"

    # ---- First vertical region statement ----
    interval_1 = serial_utils.make_interval(
        AST.Interval.Start, AST.Interval.End, 0, 0)
    body_ast_1 = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("c"),
                serial_utils.make_binary_operator(
                    serial_utils.make_field_access_expr(
                        "c"), "/", serial_utils.make_field_access_expr("b"),
                ),
                "=",
            )
        ]
    )

    vertical_region_stmt_1 = serial_utils.make_vertical_region_decl_stmt(
        body_ast_1, interval_1, AST.VerticalRegion.Forward
    )

    # ---- Second vertical region statement ----
    interval_2 = serial_utils.make_interval(
        AST.Interval.Start, AST.Interval.End, 1, 0)

    body_ast_2 = serial_utils.make_ast(
        [
            serial_utils.make_var_decl_stmt(
                serial_utils.make_type(AST.BuiltinType.Integer),
                "m",
                0,
                "=",
                serial_utils.make_expr(
                    serial_utils.make_binary_operator(
                        serial_utils.make_literal_access_expr(
                            "1.0", AST.BuiltinType.Float),
                        "/",
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr("b"),
                            "-",
                            serial_utils.make_binary_operator(
                                serial_utils.make_field_access_expr("a"),
                                "*",
                                serial_utils.make_field_access_expr(
                                    "c", [0, 0, -1]),
                            ),
                        ),
                    )
                ),
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("c"),
                serial_utils.make_binary_operator(
                    serial_utils.make_field_access_expr(
                        "c"), "*", serial_utils.make_var_access_expr("m")
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("d"),
                serial_utils.make_binary_operator(
                    serial_utils.make_binary_operator(
                        serial_utils.make_field_access_expr("d"),
                        "-",
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr("a"),
                            "*",
                            serial_utils.make_field_access_expr("d", [0, 0, -1]),
                        ),
                    ),
                    "*",
                    serial_utils.make_var_access_expr("m"),
                ),
                "=",
            ),
        ]
    )
    vertical_region_stmt_2 = serial_utils.make_vertical_region_decl_stmt(
        body_ast_2, interval_2, AST.VerticalRegion.Forward
    )

    # ---- Third vertical region statement ----
    interval_3 = serial_utils.make_interval(
        AST.Interval.Start, AST.Interval.End, 0, -1)
    body_ast_3 = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("d"),
                serial_utils.make_binary_operator(
                    serial_utils.make_field_access_expr(
                        "c"), "*", serial_utils.make_field_access_expr("d", [0, 0, 1]),
                ),
                "-=",
            )
        ]
    )

    vertical_region_stmt_3 = serial_utils.make_vertical_region_decl_stmt(
        body_ast_3, interval_3, AST.VerticalRegion.Backward
    )

    sir = serial_utils.make_sir(
        OUTPUT_FILE,
        serial_utils.GridType.Value("Cartesian"),
        [
            serial_utils.make_stencil(
                OUTPUT_NAME,
                serial_utils.make_ast(
                    [vertical_region_stmt_1, vertical_region_stmt_2, vertical_region_stmt_3]),
                [
                    serial_utils.make_field(
                        "a", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field(
                        "b", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field(
                        "c", serial_utils.make_field_dimensions_cartesian()),
                    serial_utils.make_field(
                        "d", serial_utils.make_field_dimensions_cartesian()),
                ],
            )
        ],
    )

    return sir


# ---- Unstructured cases -----
def make_unstructured_stencil_sir(name=None):
    OUTPUT_NAME = name if name is not None else "unstructured_stencil"
    OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
    interval = serial_utils.make_interval(
        AST.Interval.Start, AST.Interval.End, 0, 0)

    # create the out = in[i+1] statement
    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_unstructured_field_access_expr("out"),
                serial_utils.make_reduction_over_neighbor_expr(
                    "+",
                    serial_utils.make_literal_access_expr(
                        "1.0", AST.BuiltinType.Float),
                    serial_utils.make_unstructured_field_access_expr("in"),
                    chain=[AST.LocationType.Value('Edge'), AST.LocationType.Value('Cell')]
                ),
                "=",
            )
        ]
    )

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, AST.VerticalRegion.Forward)

    sir = serial_utils.make_sir(
        OUTPUT_FILE,
        serial_utils.GridType.Value("Unstructured"),
        [
            serial_utils.make_stencil(
                OUTPUT_NAME,
                serial_utils.make_ast([vertical_region_stmt]),
                [
                    serial_utils.make_field("in", serial_utils.make_field_dimensions_unstructured(
                        [AST.LocationType.Value('Cell')], 1)),
                    serial_utils.make_field("out", serial_utils.make_field_dimensions_unstructured(
                        [AST.LocationType.Value('Edge')], 1))
                ],
            )
        ],
    )

    return sir
