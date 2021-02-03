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

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR, AST
from dawn4py.serialization import utils as serial_utils
from google.protobuf.json_format import MessageToJson, Parse


def main(args: argparse.Namespace):
    stencil_name = "ICON_laplacian_diamond_stencil"
    gen_outputfile = f"{stencil_name}.cpp"
    sir_outputfile = f"{stencil_name}.sir"

    interval = serial_utils.make_interval(
        AST.Interval.Start, AST.Interval.End, 0, 0)

    body_ast = serial_utils.make_ast(
        [
            # fill sparse dimension vn vert using the loop concept
            serial_utils.make_loop_stmt(
                [serial_utils.make_assignment_stmt(
                    serial_utils.make_field_access_expr("vn_vert"),
                    serial_utils.make_binary_operator(
                        serial_utils.make_binary_operator(serial_utils.make_field_access_expr(
                            "u_vert", [True, 0]), "*", serial_utils.make_field_access_expr("primal_normal_x", [True, 0])),
                        "+", serial_utils.make_binary_operator(serial_utils.make_field_access_expr(
                            "v_vert", [True, 0]), "*", serial_utils.make_field_access_expr("primal_normal_y", [True, 0])),
                    ),
                    "=")],
                [AST.LocationType.Value(
                    "Edge"), AST.LocationType.Value("Cell"), AST.LocationType.Value("Vertex")]
            ),
            # dvt_tang for smagorinsky
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("dvt_tang"),
                serial_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double),
                    rhs=serial_utils.make_binary_operator(
                        serial_utils.make_binary_operator(serial_utils.make_field_access_expr(
                            "u_vert", [True, 0]), "*", serial_utils.make_field_access_expr("dual_normal_x", [True, 0])),
                        "+", serial_utils.make_binary_operator(serial_utils.make_field_access_expr(
                            "v_vert", [True, 0]), "*", serial_utils.make_field_access_expr("dual_normal_y", [True, 0])),
                    ),
                    chain=[AST.LocationType.Value("Edge"), AST.LocationType.Value(
                        "Cell"), AST.LocationType.Value("Vertex")],
                    weights=[serial_utils.make_literal_access_expr(
                        "-1.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "1.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double)]

                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("dvt_tang"), serial_utils.make_binary_operator(
                    serial_utils.make_field_access_expr("dvt_tang"), "*", serial_utils.make_field_access_expr("tangent_orientation")), "="),
            # dvt_norm for smagorinsky
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("dvt_norm"),
                serial_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double),
                    rhs=serial_utils.make_binary_operator(
                        serial_utils.make_binary_operator(serial_utils.make_field_access_expr(
                            "u_vert", [True, 0]), "*", serial_utils.make_field_access_expr("dual_normal_x", [True, 0])),
                        "+", serial_utils.make_binary_operator(serial_utils.make_field_access_expr(
                            "v_vert", [True, 0]), "*", serial_utils.make_field_access_expr("dual_normal_y", [True, 0])),
                    ),
                    chain=[AST.LocationType.Value("Edge"), AST.LocationType.Value(
                        "Cell"), AST.LocationType.Value("Vertex")],
                    weights=[serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "-1.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "1.0", AST.BuiltinType.Double)]

                ),
                "=",
            ),
            # compute smagorinsky
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("kh_smag_1"),
                serial_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double),
                    rhs=serial_utils.make_field_access_expr("vn_vert"),
                    chain=[AST.LocationType.Value("Edge"), AST.LocationType.Value(
                        "Cell"), AST.LocationType.Value("Vertex")],
                    weights=[serial_utils.make_literal_access_expr(
                        "-1.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "1.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double)]

                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("kh_smag_1"),
                serial_utils.make_binary_operator(
                    serial_utils.make_binary_operator(
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr("kh_smag_1"),
                            "*",
                            serial_utils.make_field_access_expr("tangent_orientation")),
                        "*",
                        serial_utils.make_field_access_expr("inv_primal_edge_length")), "+",
                    serial_utils.make_binary_operator(
                        serial_utils.make_field_access_expr("dvt_norm"),
                        "*",
                        serial_utils.make_field_access_expr("inv_vert_vert_length"))), "="),
            serial_utils.make_assignment_stmt(serial_utils.make_field_access_expr("kh_smag_1"),
                                           serial_utils.make_binary_operator(serial_utils.make_field_access_expr(
                                               "kh_smag_1"), "*", serial_utils.make_field_access_expr("kh_smag_1"))),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("kh_smag_2"),
                serial_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double),
                    rhs=serial_utils.make_field_access_expr("vn_vert"),
                    chain=[AST.LocationType.Value("Edge"), AST.LocationType.Value(
                        "Cell"), AST.LocationType.Value("Vertex")],
                    weights=[serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "-1.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        " 1.0", AST.BuiltinType.Double)]

                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("kh_smag_2"),
                serial_utils.make_binary_operator(
                    serial_utils.make_binary_operator(
                        serial_utils.make_field_access_expr("kh_smag_2"),
                        "*",
                        serial_utils.make_field_access_expr("inv_vert_vert_length")),
                    "+",
                    serial_utils.make_binary_operator(
                        serial_utils.make_field_access_expr("dvt_tang"),
                        "*",
                        serial_utils.make_field_access_expr("inv_primal_edge_length"))), "="),
            serial_utils.make_assignment_stmt(serial_utils.make_field_access_expr("kh_smag_2"),
                                           serial_utils.make_binary_operator(serial_utils.make_field_access_expr(
                                               "kh_smag_2"), "*", serial_utils.make_field_access_expr("kh_smag_2"))),
            # currently not able to forward a sqrt, so this is technically kh_smag**2
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("kh_smag"),
                serial_utils.make_binary_operator(serial_utils.make_field_access_expr("diff_multfac_smag"), "*",
                                               serial_utils.make_fun_call_expr("math::sqrt",
                                                                            [serial_utils.make_binary_operator(serial_utils.make_field_access_expr(
                                                                                "kh_smag_1"), "+", serial_utils.make_field_access_expr("kh_smag_2"))])),
                "="),
            # compute nabla2 using the diamond reduction
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("nabla2"),
                serial_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double),
                    rhs=serial_utils.make_binary_operator(serial_utils.make_literal_access_expr(
                        "4.0", AST.BuiltinType.Double), "*", serial_utils.make_field_access_expr("vn_vert")),
                    chain=[AST.LocationType.Value("Edge"), AST.LocationType.Value(
                        "Cell"), AST.LocationType.Value("Vertex")],
                    weights=[
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr(
                                "inv_primal_edge_length"),
                            '*',
                            serial_utils.make_field_access_expr(
                                "inv_primal_edge_length")),
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr(
                                "inv_primal_edge_length"),
                            '*',
                            serial_utils.make_field_access_expr(
                                "inv_primal_edge_length")),
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr(
                                "inv_vert_vert_length"),
                            '*',
                            serial_utils.make_field_access_expr(
                                "inv_vert_vert_length")),
                        serial_utils.make_binary_operator(
                            serial_utils.make_field_access_expr(
                                "inv_vert_vert_length"),
                            '*',
                            serial_utils.make_field_access_expr(
                                "inv_vert_vert_length")),
                    ]
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("nabla2"),
                serial_utils.make_binary_operator(
                    serial_utils.make_field_access_expr("nabla2"),
                    "-",
                    serial_utils.make_binary_operator(
                        serial_utils.make_binary_operator(serial_utils.make_binary_operator(serial_utils.make_literal_access_expr(
                            "8.0", AST.BuiltinType.Double), "*", serial_utils.make_field_access_expr("vn")), "*",
                            serial_utils.make_binary_operator(
                                serial_utils.make_field_access_expr(
                                    "inv_primal_edge_length"),
                                "*",
                                serial_utils.make_field_access_expr(
                                    "inv_primal_edge_length"))),
                        "+",
                        serial_utils.make_binary_operator(serial_utils.make_binary_operator(serial_utils.make_literal_access_expr(
                            "8.0", AST.BuiltinType.Double), "*", serial_utils.make_field_access_expr("vn")), "*",
                            serial_utils.make_binary_operator(
                                serial_utils.make_field_access_expr(
                                    "inv_vert_vert_length"),
                                "*",
                                serial_utils.make_field_access_expr(
                                    "inv_vert_vert_length"))))),
                "=")
        ]
    )

    vertical_region_stmt = serial_utils.make_vertical_region_decl_stmt(
        body_ast, interval, AST.VerticalRegion.Forward
    )

    sir = serial_utils.make_sir(
        gen_outputfile,
        AST.GridType.Value("Unstructured"),
        [
            serial_utils.make_stencil(
                stencil_name,
                serial_utils.make_ast([vertical_region_stmt]),
                [
                    serial_utils.make_field(
                        "diff_multfac_smag",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value(
                                "Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "tangent_orientation",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "inv_primal_edge_length",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "inv_vert_vert_length",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "u_vert",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "v_vert",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "primal_normal_x",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge"), AST.LocationType.Value(
                                "Cell"), AST.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "primal_normal_y",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge"), AST.LocationType.Value(
                                "Cell"), AST.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "dual_normal_x",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge"), AST.LocationType.Value(
                                "Cell"), AST.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "dual_normal_y",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge"), AST.LocationType.Value(
                                "Cell"), AST.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "vn_vert",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge"), AST.LocationType.Value(
                                "Cell"), AST.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "vn",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "dvt_tang",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "dvt_norm",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "kh_smag_1",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "kh_smag_2",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "kh_smag",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "nabla2",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                ],
            ),
        ],
    )

    # print the SIR       
    if args.verbose:
        print(MessageToJson(sir))

    # compile
    code = dawn4py.compile(sir, backend=dawn4py.CodeGenBackend.CXXNaiveIco)

    # write to file
    print(f"Writing generated code to '{gen_outputfile}'")
    with open(gen_outputfile, "w") as f:
        f.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print the generated SIR",
    )
    main(parser.parse_args())
