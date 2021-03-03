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
#include "dawn/CodeGen/Driver.h"
#include "dawn/CodeGen/Options.h"
#include "dawn/IIR/LocalVariable.h"
#include "dawn/Optimizer/Lowering.h"
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Optimizer/PassFixVersionedInputFields.h"
#include "dawn/Optimizer/PassSetStageLocationType.h"
#include "dawn/Support/Assert.h"
#include "dawn/Unittest/IIRBuilder.h"

#include "testMutator.h"

#include <fstream>

// not in use, but can be employed to rapidly inject indirected reads into existing IIR
// for debugging / get coverage quickly. example usage see below
#include "testMutator.h"

int main() {

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Cells);
    auto out_f = b.field("out_field", LocType::Cells);
    std::string stencilName = "copyCell";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Cells, b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                               b.stmt(b.assignExpr(b.at(out_f), b.at(in_f))))))));

    std::ofstream of("generated/generated_copyCell.hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    // example test mutator usage. activate for debugging only
    // injectRedirectedReads(stencilInstantiation);
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Edges);
    auto out_f = b.field("out_field", LocType::Edges);
    std::string stencilName = "copyEdge";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Edges, b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                               b.stmt(b.assignExpr(b.at(out_f), b.at(in_f))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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

    std::string stencilName = "accumulateEdgeToCell";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(out_f), b.reduceOverNeighborExpr(
                                                Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                                b.lit(0.), {LocType::Cells, LocType::Edges}))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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

    std::string stencilName = "verticalSum";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End, 1, -1,
                               b.stmt(b.assignExpr(
                                   b.at(out_f), b.binaryExpr(b.at(in_f, HOffsetType::noOffset, +1),
                                                             b.at(in_f, HOffsetType::noOffset, -1),
                                                             Op::plus))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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

    std::string stencilName = "tridiagonalSolve";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(
            b.multistage(
                LoopOrderKind::Forward,
                b.stage(LocType::Cells,
                        b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::Start, 0, 0,
                                   b.stmt(b.assignExpr(
                                       b.at(c_f), b.binaryExpr(b.at(c_f), b.at(b_f), Op::divide))),
                                   b.stmt(b.assignExpr(b.at(d_f), b.binaryExpr(b.at(d_f), b.at(b_f),
                                                                               Op::divide))))),
                b.stage(LocType::Cells,
                        b.doMethod(
                            dawn::ast::Interval::Start, dawn::ast::Interval::End, 1, 0,
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
                    b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End, 0, -1,
                               b.stmt(b.assignExpr(
                                   b.at(d_f),
                                   b.binaryExpr(b.at(d_f),
                                                b.binaryExpr(b.at(c_f),
                                                             b.at(d_f, HOffsetType::noOffset, +1),
                                                             Op::multiply),
                                                Op::minus))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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

    std::string stencilName = "diffusion";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End, b.declareVar(cnt),
                           b.stmt(b.assignExpr(
                               b.at(cnt), b.reduceOverNeighborExpr(Op::plus, b.lit(1), b.lit(0),
                                                                   {LocType::Cells, LocType::Edges,
                                                                    LocType::Cells}))),
                           b.stmt(b.assignExpr(
                               b.at(out_f),
                               b.reduceOverNeighborExpr(
                                   Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                   b.binaryExpr(b.unaryExpr(b.at(cnt), Op::minus),
                                                b.at(in_f, HOffsetType::noOffset, 0), Op::multiply),
                                   {LocType::Cells, LocType::Edges, LocType::Cells}))),
                           b.stmt(b.assignExpr(
                               b.at(out_f),
                               b.binaryExpr(b.at(in_f),
                                            b.binaryExpr(b.lit(0.1), b.at(out_f), Op::multiply),
                                            Op::plus))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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

    std::string stencilName = "gradient";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(edge_f), b.reduceOverNeighborExpr<float>(
                                                 Op::plus, b.at(cell_f, HOffsetType::withOffset, 0),
                                                 b.lit(0.), {LocType::Edges, LocType::Cells},
                                                 std::vector<float>({1., -1.})))))),
            b.stage(
                LocType::Cells,
                b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(cell_f), b.reduceOverNeighborExpr<float>(
                                                 Op::plus, b.at(edge_f, HOffsetType::withOffset, 0),
                                                 b.lit(0.), {LocType::Cells, LocType::Edges},
                                                 std::vector<float>({0.5, 0., 0., 0.5})))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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

    std::string stencilName = "diamond";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(edge_f),
                               b.reduceOverNeighborExpr(
                                   Op::plus, b.at(node_f, HOffsetType::withOffset, 0), b.lit(0.),
                                   {LocType::Edges, LocType::Cells, LocType::Vertices}))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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

    std::string stencilName = "diamondWeights";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
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

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in", LocType::Cells);
    auto out_f = b.field("out", LocType::Cells);

    std::string stencilName = "intp";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                               b.stmt(b.assignExpr(
                                   b.at(out_f),
                                   b.reduceOverNeighborExpr(
                                       Op::plus, b.at(in_f, HOffsetType::withOffset, 0), b.lit(0.),
                                       {LocType::Cells, LocType::Edges, LocType::Cells,
                                        LocType::Edges, LocType::Cells}))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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

    std::string stencilName = "sparseDimension";

    // stencil consuming a sparse dimension and a weight
    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                               b.stmt(b.assignExpr(
                                   b.at(cell_f),
                                   b.reduceOverNeighborExpr<float>(
                                       Op::plus,
                                       b.binaryExpr(b.at(edge_f, HOffsetType::withOffset, 0),
                                                    b.at(sparse_f, HOffsetType::withOffset, 0),
                                                    Op::multiply),
                                       b.lit(0.), {LocType::Cells, LocType::Edges},
                                       std::vector<float>({1., 1., 1., 1})))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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

    std::string stencilName = "sparseDimensionTwice";

    // stencil consuming a sparse dimension and a weight
    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(
                    dawn::ast::Interval::Start, dawn::ast::Interval::End,
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

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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

    std::string stencilName = "nestedSimple";

    // a nested reduction v->e->c
    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(cell_f),
                               b.reduceOverNeighborExpr(
                                   Op::plus,
                                   b.reduceOverNeighborExpr(Op::plus, b.at(vertex_f), b.lit(0.),
                                                            {LocType::Edges, LocType::Vertices}),
                                   b.lit(0.), {LocType::Cells, LocType::Edges}))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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

    std::string stencilName = "nestedWithField";

    // a nested reduction v->e->c, the edge field is also consumed "along the way"
    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(
                        dawn::ast::Interval::Start, dawn::ast::Interval::End,
                        b.stmt(b.assignExpr(
                            b.at(cell_f), b.reduceOverNeighborExpr(
                                              Op::plus,
                                              b.binaryExpr(b.at(edge_f),
                                                           b.reduceOverNeighborExpr(
                                                               Op::plus, b.at(vertex_f), b.lit(0.),
                                                               {LocType::Edges, LocType::Vertices}),
                                                           Op::plus),
                                              b.lit(0.), {LocType::Cells, LocType::Edges}))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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

    std::string stencilName = "nestedWithSparse";

    // a nested reduction v->e->c, the edge field is also consumed "along the way"
    // two additional sparse dimension fields with {c,e} and {e,v} are also introduced and consumed
    // in the outer and inner reduction, respectively
    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(
                    dawn::ast::Interval::Start, dawn::ast::Interval::End,
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

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
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

    std::string stencilName = "sparseAssignment0";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(
                    dawn::ast::Interval::Start, dawn::ast::Interval::End,
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

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
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

    std::string stencilName = "sparseAssignment1";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(
                    dawn::ast::Interval::Start, dawn::ast::Interval::End,
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

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto sparse_f = b.field("sparse", {LocType::Edges, LocType::Cells, LocType::Vertices});

    auto e_f = b.field("e", LocType::Edges);
    auto v_f = b.field("v", LocType::Vertices);

    std::string stencilName = "sparseAssignment2";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                           b.loopStmtChain(
                               b.stmt(b.assignExpr(
                                   b.at(sparse_f),
                                   b.binaryExpr(b.binaryExpr(b.lit(-4.), b.at(e_f), Op::multiply),
                                                b.at(v_f, HOffsetType::withOffset, 0), Op::plus))),
                               {LocType::Edges, LocType::Cells, LocType::Vertices}))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
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

    std::string stencilName = "sparseAssignment3";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(
                        dawn::ast::Interval::Start, dawn::ast::Interval::End,
                        b.loopStmtChain(
                            b.stmt(b.assignExpr(b.at(sparse_f),
                                                b.binaryExpr(b.at(A_f, HOffsetType::withOffset, 0),
                                                             b.at(B_f), Op::minus))),
                            {LocType::Cells, LocType::Edges, LocType::Cells, LocType::Edges,
                             LocType::Cells}))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto sparse_f = b.field("sparse", {LocType::Cells, LocType::Edges});

    auto v_f = b.field("v", LocType::Vertices);

    std::string stencilName = "sparseAssignment4";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(
                    dawn::ast::Interval::Start, dawn::ast::Interval::End,
                    b.loopStmtChain(b.stmt(b.assignExpr(b.at(sparse_f),
                                                        b.reduceOverNeighborExpr(
                                                            Op::plus, b.at(v_f), b.lit(0.),
                                                            {LocType::Edges, LocType::Vertices}))),
                                    {LocType::Cells, LocType::Edges}))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto sparse_f = b.field("sparse", {LocType::Cells, LocType::Edges});

    auto v_f = b.field("v", LocType::Vertices);
    auto c_f = b.field("c", LocType::Cells);

    std::string stencilName = "sparseAssignment5";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
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
    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
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

    std::string stencilName = "horizontalVertical";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(LocType::Edges,
                    b.doMethod(
                        dawn::ast::Interval::Start, dawn::ast::Interval::End,
                        b.stmt(b.assignExpr(
                            b.at(out1_f),
                            b.binaryExpr(b.binaryExpr(b.at(full_f), b.at(horizontal_f), Op::plus),
                                         b.at(vertical_f), Op::plus))),
                        b.stmt(b.assignExpr(
                            b.at(out2_f),
                            b.reduceOverNeighborExpr(Op::plus, b.at(sparse_horizontal_f), b.lit(.0),
                                                     {LocType::Edges, LocType::Cells}))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in = b.field("in", LocType::Cells);
    auto out = b.field("out", LocType::Cells);
    auto kidx = b.field("kidx", LocType::Cells);

    std::string stencilName = "verticalIndirecion";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End, 0, -1,
                               b.stmt(b.assignExpr(
                                   b.at(out), b.at(in, AccessType::r,
                                                   dawn::ast::Offsets{dawn::ast::unstructured,
                                                                      false, 0, "kidx"}))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto out = b.field("out", LocType::Cells);
    auto in_1 = b.field("in_1", LocType::Cells);
    auto in_2 = b.field("in_2", LocType::Cells);

    std::string stencilName = "iterationSpaceUnstructured";

    auto stencilInstantiation = b.build(
        stencilName, b.stencil(b.multistage(
                         LoopOrderKind::Parallel,
                         b.stage(LocType::Cells, Interval(3000, 4000, 0, 0),
                                 b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End, 0,
                                            0, b.stmt(b.assignExpr(b.at(out), b.at(in_1))))),
                         b.stage(LocType::Cells, Interval(2000, 3000, 0, 0),
                                 b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End, 0,
                                            0, b.stmt(b.assignExpr(b.at(out), b.at(in_2))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
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
    auto global = b.globalvar("dt", 0.5);
    std::string stencilName = "globalVar";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                           b.stmt(b.assignExpr(b.at(out_f), b.binaryExpr(b.at(global), b.at(in_f),
                                                                         Op::multiply))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto dense = b.field("dense", LocType::Edges);
    auto sparse =
        b.field("sparse", {LocType::Edges, LocType::Cells, LocType::Vertices, LocType::Edges});
    std::string stencilName = "tempField";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Edges,
                    b.doMethod(
                        dawn::ast::Interval::Start, dawn::ast::Interval::End,
                        b.stmt(b.assignExpr(b.at(dense), b.reduceOverNeighborExpr(
                                                             Op::plus, b.at(sparse), b.lit(0.),
                                                             {LocType::Edges, LocType::Cells,
                                                              LocType::Vertices, LocType::Edges}))),
                        b.loopStmtChain(b.stmt(b.assignExpr(b.at(sparse),
                                                            b.at(dense, HOffsetType::withOffset, 0),
                                                            Op::assign)),
                                        {LocType::Edges, LocType::Cells, LocType::Vertices,
                                         LocType::Edges}))))));

    dawn::PassFieldVersioning passFieldVersioning;
    dawn::PassFixVersionedInputFields passFixVersionedInputFields;
    dawn::PassSetStageLocationType passSetStageLocationType;
    passFieldVersioning.run(stencilInstantiation);
    passFixVersionedInputFields.run(stencilInstantiation);
    passSetStageLocationType.run(stencilInstantiation);

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Cells);
    auto tmp_f = b.tmpField("tmp_field", LocType::Cells);
    auto out_f = b.field("out_field", LocType::Cells);
    std::string stencilName = "tempFieldAllocation";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                           b.stmt(b.assignExpr(b.at(tmp_f), b.lit(1.))),
                           b.stmt(b.assignExpr(
                               b.at(out_f), b.binaryExpr(b.at(tmp_f), b.at(in_f), Op::plus))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Cells);
    auto tmp_f = b.tmpField("tmp_field", {LocType::Cells, LocType::Edges});
    auto out_f = b.field("out_field", LocType::Cells);
    std::string stencilName = "sparseTempFieldAllocation";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                           b.loopStmtChain(b.stmt(b.assignExpr(b.at(tmp_f), b.lit(1.))),
                                           {LocType::Cells, LocType::Edges}),
                           b.stmt(b.assignExpr(
                               b.at(out_f),
                               b.reduceOverNeighborExpr(
                                   Op::plus, b.binaryExpr(b.at(tmp_f), b.at(in_f), Op::multiply),
                                   b.lit(0.), {LocType::Cells, LocType::Edges}))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto c_f = b.field("c_field", LocType::Cells);
    auto sparse_f = b.field("sparse_field", {LocType::Cells, LocType::Edges});
    auto e_f = b.field("e_field", LocType::Edges);
    auto out_f = b.field("out_field", LocType::Cells);
    std::string stencilName = "reductionInIfConditional";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(
                        dawn::ast::Interval::Start, dawn::ast::Interval::End,
                        b.ifStmt(
                            b.binaryExpr(b.reduceOverNeighborExpr(
                                             Op::plus, b.at(e_f),
                                             b.lit(0.), {LocType::Cells, LocType::Edges}),
                                         b.lit(3.), Op::less),
                            b.stmt(b.assignExpr(
                                b.at(out_f),
                                b.reduceOverNeighborExpr(
                                    Op::plus, b.binaryExpr(b.at(sparse_f), b.lit(2.), Op::multiply),
                                    b.lit(0.), {LocType::Cells, LocType::Edges}))),
                            b.stmt(b.assignExpr(
                                b.at(out_f),
                                b.reduceOverNeighborExpr(
                                    Op::plus, b.binaryExpr(b.at(sparse_f), b.lit(4.), Op::multiply),
                                    b.lit(0.), {LocType::Cells, LocType::Edges})))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cin_f = b.field("cin_field", LocType::Cells);
    auto cout_f = b.field("cout_field", LocType::Cells);
    std::string stencilName = "reductionWithCenter";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(
                LocType::Cells,
                b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                           b.stmt(b.assignExpr(
                               b.at(cout_f),
                               b.reduceOverNeighborExpr(
                                   Op::plus, b.at(cin_f, HOffsetType::withOffset, 0), b.lit(0.),
                                   {LocType::Cells, LocType::Edges, LocType::Cells}, true))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cin_f = b.field("cin_field", LocType::Cells);
    auto cout_f = b.field("cout_field", LocType::Cells);
    auto sparse_f = b.field("sparse", {LocType::Cells, LocType::Edges, LocType::Cells},
                            /*maskK*/ true, /*include_center*/ true);
    std::string stencilName = "reductionWithCenterSparse";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                               b.stmt(b.assignExpr(
                                   b.at(cout_f),
                                   b.reduceOverNeighborExpr(
                                       Op::plus,
                                       b.binaryExpr(b.at(cin_f, HOffsetType::withOffset, 0),
                                                    b.at(sparse_f), Op::multiply),
                                       b.lit(0.), {LocType::Cells, LocType::Edges, LocType::Cells},
                                       /*include center*/ true))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto cin_f = b.field("cin_field", LocType::Cells);
    auto cout_f = b.field("cout_field", LocType::Cells);
    auto sparse_f = b.tmpField("sparse", {LocType::Cells, LocType::Edges, LocType::Cells},
                               /*maskK*/ true, /*include_center*/ true);
    std::string stencilName = "reductionAndFillWithCenterSparse";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                               b.loopStmtChain(b.stmt(b.assignExpr(b.at(sparse_f), b.lit(2.))),
                                               {LocType::Cells, LocType::Edges, LocType::Cells},
                                               /*include center*/ true),
                               b.stmt(b.assignExpr(
                                   b.at(cout_f),
                                   b.reduceOverNeighborExpr(
                                       Op::plus,
                                       b.binaryExpr(b.at(cin_f, HOffsetType::withOffset, 0),
                                                    b.at(sparse_f), Op::multiply),
                                       b.lit(0.), {LocType::Cells, LocType::Edges, LocType::Cells},
                                       /*include center*/ true))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::LocationType;

    UnstructuredIIRBuilder b;
    auto c_f = b.field("c_field", LocType::Cells);
    auto e_f = b.field("e_field", LocType::Edges);
    auto v_f = b.field("v_field", LocType::Vertices);

    std::string stencilName = "padding";

    auto stencilInstantiation = b.build(
        stencilName,
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Cells, b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                               b.stmt(b.assignExpr(b.at(c_f), b.lit(1.))))),
            b.stage(LocType::Edges, b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                                               b.stmt(b.assignExpr(b.at(e_f), b.lit(1.))))),
            b.stage(LocType::Vertices,
                    b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                               b.stmt(b.assignExpr(b.at(v_f), b.lit(1.))))))));

    std::ofstream of("generated/generated_" + stencilName + ".hpp");
    DAWN_ASSERT_MSG(of, "couldn't open output file!\n");
    dawn::codegen::Options opt;
    opt.paddingCells = 10;
    opt.paddingEdges = 20;
    opt.paddingVertices = 30;
    auto tu = dawn::codegen::run(stencilInstantiation, dawn::codegen::Backend::CXXNaiveIco, opt);
    of << dawn::codegen::generate(tu) << std::endl;
  }

  return 0;
}