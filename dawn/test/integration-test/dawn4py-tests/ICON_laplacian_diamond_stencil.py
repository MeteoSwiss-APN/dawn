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


"""Generate input for the ICON Laplacian stencil test. This is an alternative version of the diamond,
   emulating an FD stencil on a FV mesh. This is the version used in operations, since it is expected
   to offer second order convergence"""

import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils
from google.protobuf.json_format import MessageToJson, Parse


def main():
    stencil_name = "ICON_laplacian_diamond_stencil"
    gen_outputfile = f"{stencil_name}.cpp"
    sir_outputfile = f"{stencil_name}.sir"

    interval = sir_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)

    body_ast = sir_utils.make_ast(
        [
            # fill sparse dimension vn vert using the loop concept
            sir_utils.make_loop_stmt(
                [sir_utils.make_assignment_stmt(
                    sir_utils.make_field_access_expr("vn_vert"),
                    sir_utils.make_binary_operator(
                        sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
                            "u_vert", [True, 0]), "*", sir_utils.make_field_access_expr("primal_normal_x", [True, 0])),
                        "+", sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
                            "v_vert", [True, 0]), "*", sir_utils.make_field_access_expr("primal_normal_y", [True, 0])),
                    ),
                    "=")],
                [SIR.LocationType.Value(
                    "Edge"), SIR.LocationType.Value("Cell"), SIR.LocationType.Value("Vertex")]
            ),
            # dvt_tang for smagorinsky
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("dvt_tang"),
                sir_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double),
                    rhs=sir_utils.make_binary_operator(
                        sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
                            "u_vert", [True, 0]), "*", sir_utils.make_field_access_expr("dual_normal_x", [True, 0])),
                        "+", sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
                            "v_vert", [True, 0]), "*", sir_utils.make_field_access_expr("dual_normal_y", [True, 0])),
                    ),
                    chain=[SIR.LocationType.Value("Edge"), SIR.LocationType.Value(
                        "Cell"), SIR.LocationType.Value("Vertex")],
                    weights=[sir_utils.make_literal_access_expr(
                        "-1.0", SIR.BuiltinType.Double), sir_utils.make_literal_access_expr(
                        "1.0", SIR.BuiltinType.Double), sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double), sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double)]

                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("dvt_tang"), sir_utils.make_binary_operator(
                    sir_utils.make_field_access_expr("dvt_tang"), "*", sir_utils.make_field_access_expr("tangent_orientation")), "="),
            # dvt_norm for smagorinsky
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("dvt_norm"),
                sir_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double),
                    rhs=sir_utils.make_binary_operator(
                        sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
                            "u_vert", [True, 0]), "*", sir_utils.make_field_access_expr("dual_normal_x", [True, 0])),
                        "+", sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
                            "v_vert", [True, 0]), "*", sir_utils.make_field_access_expr("dual_normal_y", [True, 0])),
                    ),
                    chain=[SIR.LocationType.Value("Edge"), SIR.LocationType.Value(
                        "Cell"), SIR.LocationType.Value("Vertex")],
                    weights=[sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double), sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double), sir_utils.make_literal_access_expr(
                        "-1.0", SIR.BuiltinType.Double), sir_utils.make_literal_access_expr(
                        "1.0", SIR.BuiltinType.Double)]

                ),
                "=",
            ),
            # compute smagorinsky
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("kh_smag_1"),
                sir_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double),
                    rhs=sir_utils.make_field_access_expr("vn_vert"),
                    chain=[SIR.LocationType.Value("Edge"), SIR.LocationType.Value(
                        "Cell"), SIR.LocationType.Value("Vertex")],
                    weights=[sir_utils.make_literal_access_expr(
                        "-1.0", SIR.BuiltinType.Double), sir_utils.make_literal_access_expr(
                        "1.0", SIR.BuiltinType.Double), sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double), sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double)]

                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("kh_smag_1"),
                sir_utils.make_binary_operator(
                    sir_utils.make_binary_operator(
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr("kh_smag_1"),
                            "*",
                            sir_utils.make_field_access_expr("tangent_orientation")),
                        "*",
                        sir_utils.make_field_access_expr("inv_primal_edge_length")), "+",
                    sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("dvt_norm"),
                        "*",
                        sir_utils.make_field_access_expr("inv_vert_vert_length"))), "="),
            sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr("kh_smag_1"),
                                           sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
                                               "kh_smag_1"), "*", sir_utils.make_field_access_expr("kh_smag_1"))),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("kh_smag_2"),
                sir_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double),
                    rhs=sir_utils.make_field_access_expr("vn_vert"),
                    chain=[SIR.LocationType.Value("Edge"), SIR.LocationType.Value(
                        "Cell"), SIR.LocationType.Value("Vertex")],
                    weights=[sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double), sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double), sir_utils.make_literal_access_expr(
                        "-1.0", SIR.BuiltinType.Double), sir_utils.make_literal_access_expr(
                        " 1.0", SIR.BuiltinType.Double)]

                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("kh_smag_2"),
                sir_utils.make_binary_operator(
                    sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("kh_smag_2"),
                        "*",
                        sir_utils.make_field_access_expr("inv_vert_vert_length")),
                    "+",
                    sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("dvt_tang"),
                        "*",
                        sir_utils.make_field_access_expr("inv_primal_edge_length"))), "="),
            sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr("kh_smag_2"),
                                           sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
                                               "kh_smag_2"), "*", sir_utils.make_field_access_expr("kh_smag_2"))),
            # currently not able to forward a sqrt, so this is technically kh_smag**2
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("kh_smag"),
                sir_utils.make_binary_operator(sir_utils.make_field_access_expr("diff_multfac_smag"), "*",
                                               sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
                                                   "kh_smag_1"), "+", sir_utils.make_field_access_expr("kh_smag_2"))),
                "="),
            # compute nabla2 using the diamond reduction
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("nabla2"),
                sir_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double),
                    rhs=sir_utils.make_binary_operator(sir_utils.make_literal_access_expr(
                        "4.0", SIR.BuiltinType.Double), "*", sir_utils.make_field_access_expr("vn_vert")),
                    chain=[SIR.LocationType.Value("Edge"), SIR.LocationType.Value(
                        "Cell"), SIR.LocationType.Value("Vertex")],
                    weights=[
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr(
                                "inv_primal_edge_length"),
                            '*',
                            sir_utils.make_field_access_expr(
                                "inv_primal_edge_length")),
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr(
                                "inv_primal_edge_length"),
                            '*',
                            sir_utils.make_field_access_expr(
                                "inv_primal_edge_length")),
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr(
                                "inv_vert_vert_length"),
                            '*',
                            sir_utils.make_field_access_expr(
                                "inv_vert_vert_length")),
                        sir_utils.make_binary_operator(
                            sir_utils.make_field_access_expr(
                                "inv_vert_vert_length"),
                            '*',
                            sir_utils.make_field_access_expr(
                                "inv_vert_vert_length")),
                    ]
                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("nabla2"),
                sir_utils.make_binary_operator(
                    sir_utils.make_field_access_expr("nabla2"),
                    "-",
                    sir_utils.make_binary_operator(
                        sir_utils.make_binary_operator(sir_utils.make_binary_operator(sir_utils.make_literal_access_expr(
                            "8.0", SIR.BuiltinType.Double), "*", sir_utils.make_field_access_expr("vn")), "*",
                            sir_utils.make_binary_operator(
                                sir_utils.make_field_access_expr(
                                    "inv_primal_edge_length"),
                                "*",
                                sir_utils.make_field_access_expr(
                                    "inv_primal_edge_length"))),
                        "+",
                        sir_utils.make_binary_operator(sir_utils.make_binary_operator(sir_utils.make_literal_access_expr(
                            "8.0", SIR.BuiltinType.Double), "*", sir_utils.make_field_access_expr("vn")), "*",
                            sir_utils.make_binary_operator(
                                sir_utils.make_field_access_expr(
                                    "inv_vert_vert_length"),
                                "*",
                                sir_utils.make_field_access_expr(
                                    "inv_vert_vert_length"))))),
                "=")
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )

    sir = sir_utils.make_sir(
        gen_outputfile,
        SIR.GridType.Value("Unstructured"),
        [
            sir_utils.make_stencil(
                stencil_name,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field(
                        "diff_multfac_smag",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value(
                                "Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "tangent_orientation",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "inv_primal_edge_length",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "inv_vert_vert_length",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "u_vert",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "v_vert",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "primal_normal_x",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge"), SIR.LocationType.Value(
                                "Cell"), SIR.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "primal_normal_y",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge"), SIR.LocationType.Value(
                                "Cell"), SIR.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "dual_normal_x",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge"), SIR.LocationType.Value(
                                "Cell"), SIR.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "dual_normal_y",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge"), SIR.LocationType.Value(
                                "Cell"), SIR.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "vn_vert",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge"), SIR.LocationType.Value(
                                "Cell"), SIR.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "vn",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "dvt_tang",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "dvt_norm",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "kh_smag_1",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "kh_smag_2",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "kh_smag",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "nabla2",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                ],
            ),
        ],
    )

    # write SIR to file (for debugging purposes)
    f = open(sir_outputfile, "w")
    f.write(MessageToJson(sir))
    f.close()

    # compile
    code = dawn4py.compile(sir, backend=dawn4py.CodeGenBackend.CXXNaiveIco)

    # write to file
    print(f"Writing generated code to '{gen_outputfile}'")
    with open(gen_outputfile, "w") as f:
        f.write(code)


if __name__ == "__main__":
    main()
