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

from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils


GRID_TEST_CASES = ["copy_stencil", "hori_diff_stencil", "tridiagonal_solve_stencil"]

UNSTRUCTURED_TEST_CASES = ["unstructured_stencil"]


# ---- Grid cases -----
def make_copy_stencil_sir(name=None):
    OUTPUT_NAME = name if name is not None else "copy_stencil"
    OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"

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

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(body_ast, interval, SIR.VerticalRegion.Forward)

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        sir_utils.GridType.Value("Cartesian"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field("in", sir_utils.make_field_dimensions_cartesian()), 
                    sir_utils.make_field("out", sir_utils.make_field_dimensions_cartesian())],
            )
        ],
    )

    return sir


def make_hori_diff_stencil_sir(name=None):
    OUTPUT_NAME = name if name is not None else "hori_diff_stencil"
    OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"

    interval = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)

    # create the stencil body AST
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("lap"),
                sir_utils.make_binary_operator(
                    sir_utils.make_binary_operator(
                        sir_utils.make_literal_access_expr("-4.0", SIR.BuiltinType.Float),
                        "*",
                        sir_utils.make_field_access_expr("in"),
                    ),
                    "+",
                    sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("coeff"),
                        "*",
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr("in", [1, 0, 0]),
                            "+",
                            sir_utils.make_binary_operator(
                                sir_utils.make_field_access_expr("in", [-1, 0, 0]),
                                "+",
                                sir_utils.make_binary_operator(
                                    sir_utils.make_field_access_expr("in", [0, 1, 0]),
                                    "+",
                                    sir_utils.make_field_access_expr("in", [0, -1, 0]),
                                ),
                            ),
                        ),
                    ),
                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out"),
                sir_utils.make_binary_operator(
                    sir_utils.make_binary_operator(
                        sir_utils.make_literal_access_expr("-4.0", SIR.BuiltinType.Float),
                        "*",
                        sir_utils.make_field_access_expr("lap"),
                    ),
                    "+",
                    sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("coeff"),
                        "*",
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr("lap", [1, 0, 0]),
                            "+",
                            sir_utils.make_binary_operator(
                                sir_utils.make_field_access_expr("lap", [-1, 0, 0]),
                                "+",
                                sir_utils.make_binary_operator(
                                    sir_utils.make_field_access_expr("lap", [0, 1, 0]),
                                    "+",
                                    sir_utils.make_field_access_expr("lap", [0, -1, 0]),
                                ),
                            ),
                        ),
                    ),
                ),
                "=",
            ),
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(body_ast, interval, SIR.VerticalRegion.Forward)

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        sir_utils.GridType.Value("Cartesian"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field("in", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("out", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("coeff", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("lap", sir_utils.make_field_dimensions_cartesian(), is_temporary=True),
                ],
            )
        ],
    )

    return sir


def make_tridiagonal_solve_stencil_sir(name=None):
    OUTPUT_NAME = name if name is not None else "tridiagonal_solve_stencil"
    OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"

    # ---- First vertical region statement ----
    interval_1 = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)
    body_ast_1 = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("c"),
                sir_utils.make_binary_operator(
                    sir_utils.make_field_access_expr("c"), "/", sir_utils.make_field_access_expr("b"),
                ),
                "=",
            )
        ]
    )

    vertical_region_stmt_1 = sir_utils.make_vertical_region_decl_stmt(
        body_ast_1, interval_1, SIR.VerticalRegion.Forward
    )

    # ---- Second vertical region statement ----
    interval_2 = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 1, 0)

    body_ast_2 = sir_utils.make_ast(
        [
            sir_utils.make_var_decl_stmt(
                sir_utils.make_type(SIR.BuiltinType.Integer),
                "m",
                0,
                "=",
                sir_utils.make_expr(
                    sir_utils.make_binary_operator(
                        sir_utils.make_literal_access_expr("1.0", SIR.BuiltinType.Float),
                        "/",
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr("b"),
                            "-",
                            sir_utils.make_binary_operator(
                                sir_utils.make_field_access_expr("a"),
                                "*",
                                sir_utils.make_field_access_expr("c", [0, 0, -1]),
                            ),
                        ),
                    )
                ),
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("c"),
                sir_utils.make_binary_operator(
                    sir_utils.make_field_access_expr("c"), "*", sir_utils.make_var_access_expr("m")
                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("d"),
                sir_utils.make_binary_operator(
                    sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("d"),
                        "-",
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr("a"),
                            "*",
                            sir_utils.make_field_access_expr("d", [0, 0, -1]),
                        ),
                    ),
                    "*",
                    sir_utils.make_var_access_expr("m"),
                ),
                "=",
            ),
        ]
    )
    vertical_region_stmt_2 = sir_utils.make_vertical_region_decl_stmt(
        body_ast_2, interval_2, SIR.VerticalRegion.Forward
    )

    # ---- Third vertical region statement ----
    interval_3 = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, -1)
    body_ast_3 = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("d"),
                sir_utils.make_binary_operator(
                    sir_utils.make_field_access_expr("c"), "*", sir_utils.make_field_access_expr("d", [0, 0, 1]),
                ),
                "-=",
            )
        ]
    )

    vertical_region_stmt_3 = sir_utils.make_vertical_region_decl_stmt(
        body_ast_3, interval_3, SIR.VerticalRegion.Backward
    )

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        sir_utils.GridType.Value("Cartesian"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast([vertical_region_stmt_1, vertical_region_stmt_2, vertical_region_stmt_3]),
                [
                    sir_utils.make_field("a", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("b", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("c", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field("d", sir_utils.make_field_dimensions_cartesian()),
                ],
            )
        ],
    )

    return sir


# ---- Unstructured cases -----
def make_unstructured_stencil_sir(name=None):
    OUTPUT_NAME = name if name is not None else "unstructured_stencil"
    OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
    interval = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)

    # create the out = in[i+1] statement
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out"),
                sir_utils.make_reduction_over_neighbor_expr(
                    "+",
                    rhs = sir_utils.make_field_access_expr("in"),
                    init = sir_utils.make_literal_access_expr("1.0", SIR.BuiltinType.Float),
                    lhs_location = SIR.LocationType.Value('Edge'),
                    rhs_location = SIR.LocationType.Value('Cell')
                ),
                "=",
            )
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(body_ast, interval, SIR.VerticalRegion.Forward)

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        sir_utils.GridType.Value("Unstructured"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field("in", sir_utils.make_field_dimensions_unstructured(SIR.LocationType.Value('Cell'), 1)), 
                    sir_utils.make_field("out", sir_utils.make_field_dimensions_unstructured(SIR.LocationType.Value('Edge'), 1))
                ],
            )
        ],
    )

    return sir
