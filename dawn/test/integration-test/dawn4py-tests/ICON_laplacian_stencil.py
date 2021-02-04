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

"""Generate input for the ICON Laplacian stencil test. This is the classic Finite Volume vector Laplacian. 
   Unfortunately, it is not used in operational simulations because of bad convergence."""

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR, AST
from dawn4py.serialization import utils as serial_utils
from google.protobuf.json_format import MessageToJson, Parse


def main(args: argparse.Namespace):
    stencil_name = "ICON_laplacian_stencil"
    gen_outputfile = f"{stencil_name}.cpp"
    sir_outputfile = f"{stencil_name}.sir"

    interval = serial_utils.make_interval(AST.Interval.Start, AST.Interval.End, 0, 0)

    body_ast = serial_utils.make_ast(
        [
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("rot_vec"),
                serial_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=serial_utils.make_literal_access_expr("0.0", AST.BuiltinType.Double),
                    rhs=serial_utils.make_binary_operator(
                        serial_utils.make_field_access_expr("vec", [True, 0]),
                        "*",
                        serial_utils.make_field_access_expr("geofac_rot"),
                    ),
                    chain=[AST.LocationType.Value("Vertex"), AST.LocationType.Value("Edge")],
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("div_vec"),
                serial_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=serial_utils.make_literal_access_expr("0.0", AST.BuiltinType.Double),
                    rhs=serial_utils.make_binary_operator(
                        serial_utils.make_field_access_expr("vec", [True, 0]),
                        "*",
                        serial_utils.make_field_access_expr("geofac_div"),
                    ),
                    chain=[AST.LocationType.Value("Cell"), AST.LocationType.Value("Edge")],
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("nabla2t1_vec"),
                serial_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double),
                    rhs=serial_utils.make_field_access_expr("rot_vec", [True, 0]),
                    chain=[AST.LocationType.Value(
                        "Edge"), AST.LocationType.Value("Vertex")],
                    weights=[serial_utils.make_literal_access_expr(
                        "-1.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "1.0", AST.BuiltinType.Double)]
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("nabla2t1_vec"),
                serial_utils.make_binary_operator(
                    serial_utils.make_binary_operator(
                        serial_utils.make_field_access_expr("tangent_orientation"),
                        "*",
                        serial_utils.make_field_access_expr("nabla2t1_vec"),
                    ),
                    "/",
                    serial_utils.make_field_access_expr("primal_edge_length"),
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("nabla2t2_vec"),
                serial_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=serial_utils.make_literal_access_expr(
                        "0.0", AST.BuiltinType.Double),
                    rhs=serial_utils.make_field_access_expr("div_vec", [True, 0]),
                    chain=[AST.LocationType.Value(
                        "Edge"), AST.LocationType.Value("Cell")],
                    weights=[serial_utils.make_literal_access_expr(
                        "-1.0", AST.BuiltinType.Double), serial_utils.make_literal_access_expr(
                        "1.0", AST.BuiltinType.Double)]
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("nabla2t2_vec"),
                serial_utils.make_binary_operator(
                    serial_utils.make_field_access_expr("nabla2t2_vec"),
                    "/",
                    serial_utils.make_field_access_expr("dual_edge_length"),
                ),
                "=",
            ),
            serial_utils.make_assignment_stmt(
                serial_utils.make_field_access_expr("nabla2_vec"),
                serial_utils.make_binary_operator(
                    serial_utils.make_field_access_expr("nabla2t2_vec"),
                    "-",
                    serial_utils.make_field_access_expr("nabla2t1_vec"),
                ),
                "=",
            ),
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
                        "vec",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "div_vec",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Cell")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "rot_vec",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "nabla2t1_vec",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                        is_temporary = True
                    ),
                    serial_utils.make_field(
                        "nabla2t2_vec",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                        is_temporary = True
                    ),
                    serial_utils.make_field(
                        "nabla2_vec",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "primal_edge_length",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "dual_edge_length",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "tangent_orientation",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "geofac_rot",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Vertex"), AST.LocationType.Value("Edge")], 1
                        ),
                    ),
                    serial_utils.make_field(
                        "geofac_div",
                        serial_utils.make_field_dimensions_unstructured(
                            [AST.LocationType.Value("Cell"), AST.LocationType.Value("Edge")], 1
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
    code = dawn4py.compile(sir, groups = [], backend=dawn4py.CodeGenBackend.CXXNaiveIco)

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
