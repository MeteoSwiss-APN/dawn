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
    outputfile = "../input/test_set_stage_location_type_function_call.sir"
    interval = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)


# TODO:
#   // a nested reduction v->e->c, the edge field is also consumed "along the way"
#   auto stencil_instantiation = b.build(
#       "icon",
#       b.stencil(b.multistage(
#           dawn::iir::LoopOrderKind::Parallel,
#           b.stage(
#               LocType::Vertices,
#               b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
#                          b.stmt(b.assignExpr(
#                              b.at(rot_vec),
#                              b.reduceOverNeighborExpr(
#                                  Op::plus, b.binaryExpr(b.at(vec), b.at(geofac_rot), Op::multiply),
#                                  b.lit(0.), dawn::ast::LocationType::Vertices,
#                                  dawn::ast::LocationType::Edges))))),
#           b.stage(
#               LocType::Cells,
#               b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
#                          b.stmt(b.assignExpr(
#                              b.at(div_vec),
#                              b.reduceOverNeighborExpr(
#                                  Op::plus, b.binaryExpr(b.at(vec), b.at(geofac_div), Op::multiply),
#                                  b.lit(0.), dawn::ast::LocationType::Cells,
#                                  dawn::ast::LocationType::Edges))))),
#           b.stage(
#               LocType::Edges,
#               b.doMethod(
#                   dawn::sir::Interval::Start, dawn::sir::Interval::End,
#                   b.stmt(b.assignExpr(b.at(nabla2t1_vec),
#                                       b.reduceOverNeighborExpr(Op::plus, b.at(rot_vec), b.lit(0.),
#                                                                dawn::ast::LocationType::Edges,
#                                                                dawn::ast::LocationType::Vertices,
#                                                                std::vector<double>{-1, 1}))),
#                   b.stmt(b.assignExpr(b.at(nabla2t1_vec),
#                                       b.binaryExpr(b.binaryExpr(b.at(tangent_orientation),
#                                                                 b.at(nabla2t1_vec), Op::multiply),
#                                                    b.at(primal_edge_length), Op::divide))),
#                   b.stmt(b.assignExpr(b.at(nabla2t2_vec),
#                                       b.reduceOverNeighborExpr(Op::plus, b.at(div_vec), b.lit(0.),
#                                                                dawn::ast::LocationType::Edges,
#                                                                dawn::ast::LocationType::Cells,
#                                                                std::vector<double>{-1, 1}))),
#                   b.stmt(b.assignExpr(
#                       b.at(nabla2t2_vec),
#                       b.binaryExpr(b.at(nabla2t2_vec), b.at(dual_edge_length), Op::divide))),
#                   b.stmt(b.assignExpr(
#                       b.at(nabla2_vec),
#                       b.binaryExpr(b.at(nabla2t1_vec), b.at(nabla2t2_vec), Op::minus))))))));

    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out"),
                sir_utils.make_reduction_over_neighbor_expr(
                    "+",
                    sir_utils.make_literal_access_expr("1.0", SIR.BuiltinType.Float),
                    sir_utils.make_field_access_expr("in"),
                    lhs_location=SIR.LocationType.Value("Edge"),
                    rhs_location=SIR.LocationType.Value("Cell"),
                ),
                "=",
            )
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )
#TODO:
#   UnstructuredIIRBuilder b;
#   auto vec = b.field("vec", LocType::Edges);
#   auto div_vec = b.field("div_vec", LocType::Cells);
#   auto rot_vec = b.field("rot_vec", LocType::Vertices);
#   auto nabla2t1_vec = b.field("nabla2t1_vec", LocType::Edges);
#   auto nabla2t2_vec = b.field("nabla2t2_vec", LocType::Edges);
#   auto nabla2_vec = b.field("nabla2_vec", LocType::Edges);
#   auto primal_edge_length = b.field("primal_edge_length", LocType::Edges);
#   auto dual_edge_length = b.field("dual_edge_length", LocType::Edges);
#   auto tangent_orientation = b.field("tangent_orientation", LocType::Edges);
#   auto geofac_rot = b.field("geofac_rot", {LocType::Vertices, LocType::Edges});
#   auto geofac_div = b.field("geofac_div", {LocType::Cells, LocType::Edges});

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        SIR.GridType.Value("Unstructured"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field(
                        "in",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Cell")], 1
                        ),
                    ),
                    sir_utils.make_field(
                        "out",
                        sir_utils.make_field_dimensions_unstructured(
                            [SIR.LocationType.Value("Edge")], 1
                        ),
                    ),
                ],
            ),
        ],
    )

    f = open(outputfile, "w")
    f.write(MessageToJson(sir))
    f.close()


if __name__ == "__main__":
    main()
