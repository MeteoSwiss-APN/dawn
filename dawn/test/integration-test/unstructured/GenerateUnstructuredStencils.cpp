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

#include "dawn/AST/LocationType.h"
#include "dawn/CodeGen/CXXNaive-ico/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/Driver.h"
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/LocalVariable.h"
#include "dawn/Support/Assert.h"
#include "dawn/Unittest/IIRBuilder.h"

#include <cstring>
#include <execinfo.h>
#include <fstream>
#include <optional>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Cells);
    auto out_f = b.field("out_field", LocType::Cells);

    auto stencilInstantiation = b.build(
        "copyCell",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Cells, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                               b.stmt(b.assignExpr(b.at(out_f), b.at(in_f))))))));

    std::ofstream of("generated/generated_copyCell.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Edges);
    auto out_f = b.field("out_field", LocType::Edges);

    auto stencilInstantiation = b.build(
        "copyEdge",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Edges, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                               b.stmt(b.assignExpr(b.at(out_f), b.at(in_f))))))));

    std::ofstream of("generated/generated_copyEdge.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Edges);
    auto out_f = b.field("out_field", LocType::Cells);

    auto stencilInstantiation = b.build(
        "accumulateEdgeToCell",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(out_f), b.reduceOverNeighborExpr(
                                                Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                                b.lit(0.), {LocType::Cells, LocType::Edges}))))))));

    std::ofstream of("generated/generated_accumulateEdgeToCell.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Cells);
    auto out_f = b.field("out_field", LocType::Cells);

    auto stencilInstantiation = b.build(
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
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
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

    auto stencilInstantiation = b.build(
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
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Cells);
    auto out_f = b.field("out_field", LocType::Cells);
    auto cnt = b.localvar("cnt", dawn::BuiltinTypeID::Integer, {}, LocalVariableType::OnCells);

    auto stencilInstantiation = b.build(
        "diffusion",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(
                        dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(cnt),
                        b.stmt(b.assignExpr(b.at(cnt),
                                            b.reduceOverNeighborExpr(
                                                Op::plus, b.lit(1), b.lit(0),
                                                {LocType::Cells, LocType::Edges, LocType::Cells}))),
                        b.stmt(b.assignExpr(
                            b.at(out_f),
                            b.reduceOverNeighborExpr(
                                Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                b.binaryExpr(b.unaryExpr(b.at(cnt), Op::minus),
                                             b.at(in_f, HOffsetType::withOffset, 0), Op::multiply),
                                {LocType::Cells, LocType::Edges, LocType::Cells}))),
                        b.stmt(b.assignExpr(
                            b.at(out_f),
                            b.binaryExpr(b.at(in_f),
                                         b.binaryExpr(b.lit(0.1), b.at(out_f), Op::multiply),
                                         Op::plus))))))));

    std::ofstream of("generated/generated_diffusion.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cell_f = b.field("cell_field", LocType::Cells);
    auto edge_f = b.field("edge_field", LocType::Edges);

    auto stencilInstantiation = b.build(
        "gradient",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(edge_f), b.reduceOverNeighborExpr<float>(
                                                 Op::plus, b.at(cell_f, HOffsetType::withOffset, 0),
                                                 b.lit(0.), {LocType::Edges, LocType::Cells},
                                                 std::vector<float>({1., -1.})))))),
            b.stage(
                LocType::Cells,
                b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(cell_f), b.reduceOverNeighborExpr<float>(
                                                 Op::plus, b.at(edge_f, HOffsetType::withOffset, 0),
                                                 b.lit(0.), {LocType::Cells, LocType::Edges},
                                                 std::vector<float>({0.5, 0., 0., 0.5})))))))));

    std::ofstream of("generated/generated_gradient.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto edge_f = b.field("edge_field", LocType::Edges);
    auto node_f = b.field("vertex_field", LocType::Vertices);

    auto stencilInstantiation = b.build(
        "diamond",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(edge_f),
                               b.reduceOverNeighborExpr(
                                   Op::plus, b.at(node_f, HOffsetType::withOffset, 0), b.lit(0.),
                                   {LocType::Edges, LocType::Cells, LocType::Vertices}))))))));

    std::ofstream of("generated/generated_diamond.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto out_f = b.field("out", LocType::Edges);
    auto inv_edge_length_f = b.field("inv_edge_length", LocType::Edges);
    auto inv_vert_length_f = b.field("inv_vert_length", LocType::Edges);
    auto in_f = b.field("in", LocType::Vertices);

    auto stencil_instantiation = b.build(
        "diamondWeights",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(out_f),
                               b.reduceOverNeighborExpr(
                                   Op::plus, b.at(in_f, HOffsetType::withOffset, 0), b.lit(0.),
                                   {LocType::Edges, LocType::Cells, LocType::Vertices},
                                   {b.binaryExpr(b.at(inv_edge_length_f), b.at(inv_edge_length_f),
                                                 Op::multiply),
                                    b.binaryExpr(b.at(inv_edge_length_f), b.at(inv_edge_length_f),
                                                 Op::multiply),
                                    b.binaryExpr(b.at(inv_vert_length_f), b.at(inv_vert_length_f),
                                                 Op::multiply),
                                    b.binaryExpr(b.at(inv_vert_length_f), b.at(inv_vert_length_f),
                                                 Op::multiply)}))))))));

    std::ofstream of("generated/generated_diamondWeights.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencil_instantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in", LocType::Cells);
    auto out_f = b.field("out", LocType::Cells);

    auto stencilInstantiation = b.build(
        "intp",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                               b.stmt(b.assignExpr(
                                   b.at(out_f),
                                   b.reduceOverNeighborExpr(
                                       Op::plus, b.at(in_f, HOffsetType::withOffset, 0), b.lit(0.),
                                       {LocType::Cells, LocType::Edges, LocType::Cells,
                                        LocType::Edges, LocType::Cells}))))))));

    std::ofstream of("generated/generated_intp.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cell_f = b.field("cell_field", LocType::Cells);
    auto edge_f = b.field("edge_field", LocType::Edges);
    auto sparse_f = b.field("sparse_dim", {LocType::Cells, LocType::Edges});

    // stencil consuming a sparse dimension and a weight
    auto stencilInstantiation = b.build(
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
                                       b.lit(0.), {LocType::Cells, LocType::Edges},
                                       std::vector<float>({1., 1., 1., 1})))))))));

    std::ofstream of("generated/generated_sparseDimension.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cell_f = b.field("cell_field", LocType::Cells);
    auto edge_f = b.field("edge_field", LocType::Edges);
    auto sparse_f = b.field("sparse_dim", {LocType::Cells, LocType::Edges});

    // stencil consuming a sparse dimension and a weight
    auto stencilInstantiation = b.build(
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
                            b.lit(0.), {LocType::Cells, LocType::Edges},
                            std::vector<float>({1., 1., 1., 1})))),
                    b.stmt(b.assignExpr(
                        b.at(cell_f),
                        b.reduceOverNeighborExpr<float>(
                            Op::plus,
                            b.binaryExpr(b.at(edge_f, HOffsetType::withOffset, 0),
                                         b.at(sparse_f, HOffsetType::withOffset, 0), Op::multiply),
                            b.lit(0.), {LocType::Cells, LocType::Edges},
                            std::vector<float>({1., 1., 1., 1})))))))));

    std::ofstream of("generated/generated_sparseDimensionTwice.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cell_f = b.field("cell_field", LocType::Cells);
    auto edge_f = b.field("edge_field", LocType::Edges);
    auto vertex_f = b.field("vertex_field", LocType::Vertices);

    // a nested reduction v->e->c
    auto stencilInstantiation = b.build(
        "nestedSimple",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(cell_f),
                               b.reduceOverNeighborExpr(
                                   Op::plus,
                                   b.reduceOverNeighborExpr(Op::plus, b.at(vertex_f), b.lit(0.),
                                                            {LocType::Edges, LocType::Vertices}),
                                   b.lit(0.), {LocType::Cells, LocType::Edges}))))))));

    std::ofstream of("generated/generated_NestedSimple.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cell_f = b.field("cell_field", LocType::Cells);
    auto edge_f = b.field("edge_field", LocType::Edges);
    auto vertex_f = b.field("vertex_field", LocType::Vertices);

    // a nested reduction v->e->c, the edge field is also consumed "along the way"
    auto stencilInstantiation = b.build(
        "nestedWithField",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(
                        dawn::sir::Interval::Start, dawn::sir::Interval::End,
                        b.stmt(b.assignExpr(
                            b.at(cell_f), b.reduceOverNeighborExpr(
                                              Op::plus,
                                              b.binaryExpr(b.at(edge_f),
                                                           b.reduceOverNeighborExpr(
                                                               Op::plus, b.at(vertex_f), b.lit(0.),
                                                               {LocType::Edges, LocType::Vertices}),
                                                           Op::plus),
                                              b.lit(0.), {LocType::Cells, LocType::Edges}))))))));

    std::ofstream of("generated/generated_NestedWithField.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
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
                                    b.lit(0.), {LocType::Edges, LocType::Vertices}),
                                Op::plus),
                            b.lit(0.), {LocType::Cells, LocType::Edges}))))))));

    // Code generation deactivated for the reasons stated above
    std::ofstream of("generated/generated_NestedSparse.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencil_instantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto vn_f = b.field("vn", {LocType::Edges, LocType::Cells, LocType::Vertices});

    auto uVert_f = b.field("uVert", LocType::Vertices);
    auto vVert_f = b.field("vVert", LocType::Vertices);
    auto nx_f = b.field("nx", LocType::Vertices);
    auto ny_f = b.field("ny", LocType::Vertices);

    auto stencil_instantiation = b.build(
        "sparseAssignment0",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(
                    dawn::sir::Interval::Start, dawn::sir::Interval::End,
                    b.loopStmtChain(
                        b.stmt(b.assignExpr(
                            b.at(vn_f),
                            b.binaryExpr(
                                b.binaryExpr(b.at(uVert_f, HOffsetType::withOffset, 0),
                                             b.at(nx_f, HOffsetType::withOffset, 0), Op::multiply),
                                b.binaryExpr(b.at(vVert_f, HOffsetType::withOffset, 0),
                                             b.at(ny_f, HOffsetType::withOffset, 0), Op::multiply),
                                Op::plus))),
                        {LocType::Edges, LocType::Cells, LocType::Vertices}))))));

    std::ofstream of("generated/generated_SparseAssignment0.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencil_instantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto vn_f = b.field("vn", {LocType::Edges, LocType::Cells, LocType::Vertices});

    auto uVert_f = b.field("uVert", LocType::Vertices);
    auto vVert_f = b.field("vVert", LocType::Vertices);
    auto nx_f = b.field("nx", {LocType::Edges, LocType::Cells, LocType::Vertices});
    auto ny_f = b.field("ny", {LocType::Edges, LocType::Cells, LocType::Vertices});

    auto stencil_instantiation = b.build(
        "sparseAssignment1",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(
                    dawn::sir::Interval::Start, dawn::sir::Interval::End,
                    b.loopStmtChain(
                        b.stmt(b.assignExpr(
                            b.at(vn_f),
                            b.binaryExpr(
                                b.binaryExpr(b.at(uVert_f, HOffsetType::withOffset, 0),
                                             b.at(nx_f, HOffsetType::withOffset, 0), Op::multiply),
                                b.binaryExpr(b.at(vVert_f, HOffsetType::withOffset, 0),
                                             b.at(ny_f, HOffsetType::withOffset, 0), Op::multiply),
                                Op::plus))),
                        {LocType::Edges, LocType::Cells, LocType::Vertices}))))));

    std::ofstream of("generated/generated_SparseAssignment1.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencil_instantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto sparse_f = b.field("sparse", {LocType::Edges, LocType::Cells, LocType::Vertices});

    auto e_f = b.field("e", LocType::Edges);
    auto v_f = b.field("v", LocType::Vertices);

    auto stencil_instantiation = b.build(
        "sparseAssignment2",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                           b.loopStmtChain(
                               b.stmt(b.assignExpr(
                                   b.at(sparse_f),
                                   b.binaryExpr(b.binaryExpr(b.lit(-4.), b.at(e_f), Op::multiply),
                                                b.at(v_f, HOffsetType::withOffset, 0), Op::plus))),
                               {LocType::Edges, LocType::Cells, LocType::Vertices}))))));

    std::ofstream of("generated/generated_SparseAssignment2.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencil_instantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto sparse_f = b.field(
        "sparse", {LocType::Cells, LocType::Edges, LocType::Cells, LocType::Edges, LocType::Cells});

    auto A_f = b.field("A", LocType::Cells);
    auto B_f = b.field("B", LocType::Cells);

    auto stencil_instantiation = b.build(
        "sparseAssignment3",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(
                        dawn::sir::Interval::Start, dawn::sir::Interval::End,
                        b.loopStmtChain(
                            b.stmt(b.assignExpr(b.at(sparse_f),
                                                b.binaryExpr(b.at(A_f, HOffsetType::withOffset, 0),
                                                             b.at(B_f), Op::minus))),
                            {LocType::Cells, LocType::Edges, LocType::Cells, LocType::Edges,
                             LocType::Cells}))))));

    std::ofstream of("generated/generated_SparseAssignment3.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencil_instantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto sparse_f = b.field("sparse", {LocType::Cells, LocType::Edges});

    auto v_f = b.field("v", LocType::Vertices);

    auto stencil_instantiation = b.build(
        "sparseAssignment4",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(
                    dawn::sir::Interval::Start, dawn::sir::Interval::End,
                    b.loopStmtChain(b.stmt(b.assignExpr(b.at(sparse_f),
                                                        b.reduceOverNeighborExpr(
                                                            Op::plus, b.at(v_f), b.lit(0.),
                                                            {LocType::Edges, LocType::Vertices}))),
                                    {LocType::Cells, LocType::Edges}))))));

    std::ofstream of("generated/generated_SparseAssignment4.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencil_instantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto sparse_f = b.field("sparse", {LocType::Cells, LocType::Edges});

    auto v_f = b.field("v", LocType::Vertices);
    auto c_f = b.field("c", LocType::Cells);

    auto stencil_instantiation = b.build(
        "sparseAssignment5",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                               b.loopStmtChain(
                                   b.stmt(b.assignExpr(
                                       b.at(sparse_f),
                                       b.reduceOverNeighborExpr(
                                           Op::plus,
                                           b.binaryExpr(b.at(v_f),
                                                        b.reduceOverNeighborExpr(
                                                            Op::plus, b.at(c_f), b.lit(0.),
                                                            {LocType::Vertices, LocType::Cells}),
                                                        Op::multiply),
                                           b.lit(0.), {LocType::Edges, LocType::Vertices}))),
                                   {LocType::Cells, LocType::Edges}))))));

    std::ofstream of("generated/generated_SparseAssignment5.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencil_instantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;

    auto horizontal_f = b.field("horizontal", {LocType::Edges}, false);
    auto sparse_horizontal_f =
        b.field("horizontal_sparse", {LocType::Edges, LocType::Cells}, false);
    auto vertical_f = b.vertical_field("vertical");
    auto full_f = b.field("full", LocType::Edges);
    auto out1_f = b.field("out1", LocType::Edges);
    auto out2_f = b.field("out2", LocType::Edges);

    auto stencil_instantiation = b.build(
        "horizontalVertical",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Edges,
                    b.doMethod(
                        dawn::sir::Interval::Start, dawn::sir::Interval::End,
                        b.stmt(b.assignExpr(
                            b.at(out1_f),
                            b.binaryExpr(b.binaryExpr(b.at(full_f), b.at(horizontal_f), Op::plus),
                                         b.at(vertical_f), Op::plus))),
                        b.stmt(b.assignExpr(
                            b.at(out2_f),
                            b.reduceOverNeighborExpr(Op::plus, b.at(sparse_horizontal_f), b.lit(.0),
                                                     {LocType::Edges, LocType::Cells}))))))));

    std::ofstream of("generated/generated_horizontal_vertical.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencil_instantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  return 0;
}
