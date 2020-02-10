//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "dawn/CodeGen/CXXNaive-ico/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/Unittest/IIRBuilder.h"

#include <cstring>
#include <fstream>
#include <optional>

template <typename CG>
void dump(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
  dawn::DiagnosticsEngine diagnostics;
  CG generator(ctx, diagnostics, 0);
  auto tu = generator.generateCode();

  std::ostringstream ss;
  for(auto const& macroDefine : tu->getPPDefines())
    ss << macroDefine << "\n";

  ss << tu->getGlobals();
  for(auto const& s : tu->getStencils())
    ss << s.second;
  os << ss.str();
}

int main() {

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Cells);
    auto out_f = b.field("out_field", LocType::Cells);

    auto stencil_instantiation = b.build(
        "copyCell", b.stencil(b.multistage(
                        LoopOrderKind::Parallel,
                        b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                           b.stmt(b.assignExpr(b.at(out_f), b.at(in_f))))))));

    std::ofstream of("generated/generated_copyCell.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Edges);
    auto out_f = b.field("out_field", LocType::Edges);

    auto stencil_instantiation = b.build(
        "copyEdge",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Edges, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                               b.stmt(b.assignExpr(b.at(out_f), b.at(in_f))))))));

    std::ofstream of("generated/generated_copyEdge.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Edges);
    auto out_f = b.field("out_field", LocType::Cells);

    auto stencil_instantiation =
        b.build("accumulateEdgeToCell",
                b.stencil(b.multistage(
                    LoopOrderKind::Parallel,
                    b.stage(b.doMethod(
                        dawn::sir::Interval::Start, dawn::sir::Interval::End,
                        b.stmt(b.assignExpr(b.at(out_f),
                                            b.reduceOverNeighborExpr(
                                                Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                                b.lit(0.), LocType::Cells, LocType::Edges))))))));

    std::ofstream of("generated/generated_accumulateEdgeToCell.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Cells);
    auto out_f = b.field("out_field", LocType::Cells);

    auto stencil_instantiation = b.build(
        "verticalSum",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End, 1, -1,
                               b.stmt(b.assignExpr(
                                   b.at(out_f), b.binaryExpr(b.at(in_f, HOffsetType::noOffset, +1),
                                                             b.at(in_f, HOffsetType::noOffset, -1),
                                                             Op::plus))))))));

    std::ofstream of("generated/generated_verticalSum.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto a_f = b.field("a", LocType::Cells);
    auto b_f = b.field("b", LocType::Cells);
    auto c_f = b.field("c", LocType::Cells);
    auto d_f = b.field("d", LocType::Cells);
    auto m_var = b.localvar("m");

    auto stencil_instantiation = b.build(
        "tridiagonalSolve",
        b.stencil(
            b.multistage(
                LoopOrderKind::Forward,
                b.stage(LocType::Cells,
                        b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::Start, 0, 0,
                                   b.stmt(b.assignExpr(
                                       b.at(c_f), b.binaryExpr(b.at(c_f), b.at(b_f), Op::divide))),
                                   b.stmt(b.assignExpr(
                                       b.at(d_f), b.binaryExpr(b.at(d_f), b.at(b_f), Op::divide)))),
                        b.doMethod(
                            dawn::sir::Interval::Start, dawn::sir::Interval::End, 1, 0,
                            b.declareVar(m_var),
                            b.stmt(b.assignExpr(
                                b.at(m_var),
                                b.binaryExpr(
                                    b.lit(1.),
                                    b.binaryExpr(b.at(b_f),
                                                 b.binaryExpr(b.at(a_f),
                                                              b.at(c_f, HOffsetType::noOffset, -1),
                                                              Op::multiply),
                                                 Op::minus),
                                    Op::divide))),
                            b.stmt(b.assignExpr(
                                b.at(c_f), b.binaryExpr(b.at(c_f), b.at(m_var), Op::multiply))),
                            b.stmt(b.assignExpr(
                                b.at(d_f),
                                b.binaryExpr(
                                    b.binaryExpr(b.at(d_f),
                                                 b.binaryExpr(b.at(a_f),
                                                              b.at(d_f, HOffsetType::noOffset, -1),
                                                              Op::multiply),
                                                 Op::minus),
                                    b.at(m_var), Op::multiply)))))),
            b.multistage(
                LoopOrderKind::Backward,
                b.stage(
                    LocType::Cells,
                    b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End, 0, -1,
                               b.stmt(b.assignExpr(
                                   b.at(d_f),
                                   b.binaryExpr(b.at(d_f),
                                                b.binaryExpr(b.at(c_f),
                                                             b.at(d_f, HOffsetType::noOffset, +1),
                                                             Op::multiply),
                                                Op::minus))))))));

    std::ofstream of("generated/generated_verticalSolver.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Cells);
    auto out_f = b.field("out_field", LocType::Cells);
    auto cnt = b.localvar("cnt", dawn::BuiltinTypeID::Integer);

    auto stencil_instantiation = b.build(
        "diffusion",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(b.doMethod(
                dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(cnt),
                b.stmt(b.assignExpr(b.at(cnt),
                                    b.reduceOverNeighborExpr(Op::plus, b.lit(1), b.lit(0),
                                                             dawn::ast::LocationType::Cells,
                                                             dawn::ast::LocationType::Cells))),
                b.stmt(b.assignExpr(
                    b.at(out_f),
                    b.reduceOverNeighborExpr(
                        Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                        b.binaryExpr(b.unaryExpr(b.at(cnt), Op::minus),
                                     b.at(in_f, HOffsetType::withOffset, 0), Op::multiply),
                        dawn::ast::LocationType::Cells, dawn::ast::LocationType::Cells))),
                b.stmt(b.assignExpr(
                    b.at(out_f),
                    b.binaryExpr(b.at(in_f), b.binaryExpr(b.lit(0.1), b.at(out_f), Op::multiply),
                                 Op::plus))))))));

    std::ofstream of("generated/generated_diffusion.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cell_f = b.field("cell_field", LocType::Cells);
    auto edge_f = b.field("edge_field", LocType::Edges);

    auto stencil_instantiation = b.build(
        "gradient",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(edge_f),
                               b.reduceOverNeighborExpr<float>(
                                   Op::plus, b.at(cell_f, HOffsetType::withOffset, 0), b.lit(0.),
                                   dawn::ast::LocationType::Edges, dawn::ast::LocationType::Cells,
                                   std::vector<float>({1., -1.})))))),
            b.stage(
                LocType::Cells,
                b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(cell_f),
                               b.reduceOverNeighborExpr<float>(
                                   Op::plus, b.at(edge_f, HOffsetType::withOffset, 0), b.lit(0.),
                                   dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges,
                                   std::vector<float>({0.5, 0., 0., 0.5})))))))));

    std::ofstream of("generated/generated_gradient.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cell_f = b.field("cell_field", LocType::Cells);
    auto edge_f = b.field("edge_field", LocType::Edges);
    auto sparse_f = b.field("sparse_dim", {LocType::Cells, LocType::Edges});

    // stencil consuming a sparse dimension and a weight
    auto stencil_instantiation = b.build(
        "sparseDimension",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                               b.stmt(b.assignExpr(
                                   b.at(cell_f),
                                   b.reduceOverNeighborExpr<float>(
                                       Op::plus,
                                       b.binaryExpr(b.at(edge_f, HOffsetType::withOffset, 0),
                                                    b.at(sparse_f, HOffsetType::withOffset, 0),
                                                    Op::multiply),
                                       b.lit(0.), dawn::ast::LocationType::Cells,
                                       dawn::ast::LocationType::Edges,
                                       std::vector<float>({1., 1., 1., 1})))))))));

    std::ofstream of("generated/generated_sparseDimension.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cell_f = b.field("cell_field", LocType::Cells);
    auto edge_f = b.field("edge_field", LocType::Edges);
    auto sparse_f = b.field("sparse_dim", {LocType::Cells, LocType::Edges});

    // stencil consuming a sparse dimension and a weight
    auto stencil_instantiation = b.build(
        "sparseDimensionTwice",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(
                    dawn::sir::Interval::Start, dawn::sir::Interval::End,
                    b.stmt(b.assignExpr(
                        b.at(cell_f),
                        b.reduceOverNeighborExpr<float>(
                            Op::plus,
                            b.binaryExpr(b.at(edge_f, HOffsetType::withOffset, 0),
                                         b.at(sparse_f, HOffsetType::withOffset, 0), Op::multiply),
                            b.lit(0.), dawn::ast::LocationType::Cells,
                            dawn::ast::LocationType::Edges, std::vector<float>({1., 1., 1., 1})))),
                    b.stmt(b.assignExpr(
                        b.at(cell_f),
                        b.reduceOverNeighborExpr<float>(
                            Op::plus,
                            b.binaryExpr(b.at(edge_f, HOffsetType::withOffset, 0),
                                         b.at(sparse_f, HOffsetType::withOffset, 0), Op::multiply),
                            b.lit(0.), dawn::ast::LocationType::Cells,
                            dawn::ast::LocationType::Edges,
                            std::vector<float>({1., 1., 1., 1})))))))));

    std::ofstream of("generated/generated_sparseDimensionTwice.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cell_f = b.field("cell_field", LocType::Cells);
    auto edge_f = b.field("edge_field", LocType::Edges);
    auto vertex_f = b.field("vertex_field", LocType::Vertices);

    // a nested reduction v->e->c
    auto stencil_instantiation = b.build(
        "nestedSimple",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                               b.stmt(b.assignExpr(
                                   b.at(cell_f),
                                   b.reduceOverNeighborExpr(
                                       Op::plus,
                                       b.reduceOverNeighborExpr(Op::plus, b.at(vertex_f), b.lit(0.),
                                                                dawn::ast::LocationType::Edges,
                                                                dawn::ast::LocationType::Vertices),
                                       b.lit(0.), dawn::ast::LocationType::Cells,
                                       dawn::ast::LocationType::Edges))))))));

    std::ofstream of("generated/generated_NestedSimple.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cell_f = b.field("cell_field", LocType::Cells);
    auto edge_f = b.field("edge_field", LocType::Edges);
    auto vertex_f = b.field("vertex_field", LocType::Vertices);

    // a nested reduction v->e->c, the edge field is also consumed "along the way"
    auto stencil_instantiation =
        b.build("nestedWithField",
                b.stencil(b.multistage(
                    dawn::iir::LoopOrderKind::Parallel,
                    b.stage(LocType::Cells,
                            b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                       b.stmt(b.assignExpr(
                                           b.at(cell_f),
                                           b.reduceOverNeighborExpr(
                                               Op::plus,
                                               b.binaryExpr(b.at(edge_f),
                                                            b.reduceOverNeighborExpr(
                                                                Op::plus, b.at(vertex_f), b.lit(0.),
                                                                dawn::ast::LocationType::Edges,
                                                                dawn::ast::LocationType::Vertices),
                                                            Op::plus),
                                               b.lit(0.), dawn::ast::LocationType::Cells,
                                               dawn::ast::LocationType::Edges))))))));

    std::ofstream of("generated/generated_NestedWithField.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cell_f = b.field("cell_field", LocType::Cells);
    auto edge_f = b.field("edge_field", LocType::Edges);
    auto vertex_f = b.field("vertex_field", LocType::Vertices);

    auto sparse_ce_f = b.field("ce_sparse", {LocType::Cells, LocType::Edges});
    auto sparse_ev_f = b.field("ev_sparse", {LocType::Edges, LocType::Vertices});

    // a nested reduction v->e->c, the edge field is also consumed "along the way"
    // two additional sparse dimension fields with {c,e} and {e,v} are also introduced and consumed
    // in the outer and inner reduction, respectively
    //
    // currently this IIR is rejected. The type of the inner sparse dimension needs to be clarified
    // options include:
    //      - this IIR is ok, some modifications with regard to code generation and/or the interface
    //      are needed
    //      - this IIR is not ok. The stage fixes the dense part of the sparse dimension, the
    //      correct type of the inner sparse dimension is {C, E->V}
    auto stencil_instantiation = b.build(
        "nestedWithSparse",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(
                    dawn::sir::Interval::Start, dawn::sir::Interval::End,
                    b.stmt(b.assignExpr(
                        b.at(cell_f),
                        b.reduceOverNeighborExpr(
                            Op::plus,
                            b.binaryExpr(
                                b.binaryExpr(b.at(edge_f), b.at(sparse_ce_f), Op::multiply),
                                b.reduceOverNeighborExpr(
                                    Op::plus,
                                    b.binaryExpr(b.at(vertex_f), b.at(sparse_ev_f), Op::multiply),
                                    b.lit(0.), dawn::ast::LocationType::Edges,
                                    dawn::ast::LocationType::Vertices),
                                Op::plus),
                            b.lit(0.), dawn::ast::LocationType::Cells,
                            dawn::ast::LocationType::Edges))))))));

    // Code generation deactivated for the reasons stated above
    // std::ofstream of("generated/generated_NestedSparse.hpp");
    // dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    // of.close();
  }

  return 0;
}
