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

"""Generate input for the ICON Laplacian stencil test"""

import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils
from google.protobuf.json_format import MessageToJson, Parse


def main():
    stencil_name = "ICON_laplacian_stencil"
    gen_outputfile = f"{stencil_name}.cpp"
    sir_outputfile = f"{stencil_name}.sir"

    interval = sir_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)

    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("rot_vec"),
                sir_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double),
                    rhs=sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("vec"),
                        "*",
                        sir_utils.make_field_access_expr("geofac_rot")),
                    chain=[SIR.LocationType.Value(
                        "Vertex"), SIR.LocationType.Value("Edge")]
                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("div_vec"),
                sir_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double),
                    rhs=sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("vec"),
                        "*",
                        sir_utils.make_field_access_expr("geofac_div")),
                    chain=[SIR.LocationType.Value(
                        "Cell"), SIR.LocationType.Value("Edge")]
                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("nabla2t1_vec"),
                sir_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double),
                    rhs=sir_utils.make_field_access_expr("rot_vec"),
                    chain=[SIR.LocationType.Value(
                        "Edge"), SIR.LocationType.Value("Vertex")],
                    weights=sir_utils.make_weights([-1.0, 1.0])
                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("nabla2t1_vec"),
                sir_utils.make_binary_operator(
                    sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr(
                            "tangent_orientation"),
                        "*",
                        sir_utils.make_field_access_expr("nabla2t1_vec")),
                    "/",
                    sir_utils.make_field_access_expr("primal_edge_length")),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("nabla2t2_vec"),
                sir_utils.make_reduction_over_neighbor_expr(
                    op="+",
                    init=sir_utils.make_literal_access_expr(
                        "0.0", SIR.BuiltinType.Double),
                    rhs=sir_utils.make_field_access_expr("div_vec"),
                    chain=[SIR.LocationType.Value(
                        "Edge"), SIR.LocationType.Value("Cell")],
                    weights=sir_utils.make_weights([-1.0, 1.0])
                ),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("nabla2t2_vec"),
                sir_utils.make_binary_operator(
                    sir_utils.make_field_access_expr("nabla2t2_vec"),
                    "/",
                    sir_utils.make_field_access_expr("dual_edge_length")),
                "=",
            ),
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("nabla2_vec"),
                sir_utils.make_binary_operator(
                    sir_utils.make_field_access_expr("nabla2t2_vec"),
                    "-",
                    sir_utils.make_field_access_expr("nabla2t1_vec")),
                "=",
            ),
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
                        "vec",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "div_vec",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "rot_vec",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Vertex")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "nabla2t1_vec",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "nabla2t2_vec",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "nabla2_vec",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "primal_edge_length",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "dual_edge_length",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "tangent_orientation",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "geofac_rot",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value(
                                "Vertex"), SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "geofac_div",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value(
                                "Cell"), SIR.LocationType.Value("Edge")], 1
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
    code = dawn4py.compile(sir, backend="c++-naive-ico")

    # write to file
    print(f"Writing generated code to '{gen_outputfile}'")
    with open(gen_outputfile, "w") as f:
        f.write(code)


if __name__ == "__main__":
    main()
