//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "dawn-c/Compiler.h"
#include "dawn-c/TranslationUnit.h"
#include "dawn/CodeGen/CXXNaive-ico/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "dawn/Unittest/IIRBuilder.h"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>

namespace {

static void freeCharArray(char** array, int size) {
  for(int i = 0; i < size; ++i)
    std::free(array[i]);
  std::free(array);
}

TEST(CompilerTest, CompileEmptySIR) {
  std::string sir;
  dawnTranslationUnit_t* TU = dawnCompile(sir.data(), sir.size(), nullptr);

  EXPECT_EQ(dawnTranslationUnitGetStencil(TU, "invalid"), nullptr);

  char** ppDefines;
  int size;
  dawnTranslationUnitGetPPDefines(TU, &ppDefines, &size);
  EXPECT_NE(size, 0);
  EXPECT_NE(ppDefines, nullptr);

  freeCharArray(ppDefines, size);
  dawnTranslationUnitDestroy(TU);
}

TEST(CompilerTest, CompileCopyStencil) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto in_f = b.field("in_field", FieldType::ijk);
  auto out_f = b.field("out_field", FieldType::ijk);

  auto stencil_instantiation =
      b.build("generated",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.block(b.stmt(b.assignExpr(b.at(out_f), b.at(in_f)))))))));
  std::ofstream of("/dev/null");
  dawn::CompilerUtil::dumpNaive(of, stencil_instantiation);
}

TEST(CompilerTest, DISABLED_CodeGenSumEdgeToCells) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Edges);
  auto out_f = b.field("out_field", LocType::Cells);
  auto cnt = b.localvar("cnt", dawn::BuiltinTypeID::Integer);

  auto stencil_instantiation = b.build(
      "generated",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Edges, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                             b.stmt(b.assignExpr(b.at(in_f), b.lit(10))))),
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.stmt(b.assignExpr(
                                 b.at(out_f), b.reduceOverNeighborExpr(
                                                  Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                                  b.lit(0.), LocType::Cells, LocType::Edges))))))));

  std::ofstream of("prototype/generated_copyEdgeToCell.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dawn::CompilerUtil::dumpNaiveIco(of, stencil_instantiation);
}

TEST(CompilerTest, DISABLED_SumVertical) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Cells);
  auto out_f = b.field("out_field", LocType::Cells);

  auto stencil_instantiation = b.build(
      "generated",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Cells,
                  b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End, 1, -1,
                             b.stmt(b.assignExpr(b.at(out_f),
                                                 b.binaryExpr(b.at(in_f, HOffsetType::noOffset, +1),
                                                              b.at(in_f, HOffsetType::noOffset, -1),
                                                              Op::plus))))))));

  std::ofstream of("prototype/generated_verticalSum.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dawn::CompilerUtil::dumpNaiveIco(of, stencil_instantiation);
}

TEST(CompilerTest, DISABLED_CodeGenDiffusion) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Cells);
  auto out_f = b.field("out_field", LocType::Cells);
  auto cnt = b.localvar("cnt", dawn::BuiltinTypeID::Integer);

  auto stencil_instantiation = b.build(
      "generated",
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
              b.stmt(b.assignExpr(b.at(out_f),
                                  b.binaryExpr(b.at(in_f),
                                               b.binaryExpr(b.lit(0.1), b.at(out_f), Op::multiply),
                                               Op::plus))))))));

  std::ofstream of("prototype/generated_Diffusion.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dawn::CompilerUtil::dumpNaiveIco(of, stencil_instantiation);
}

TEST(CompilerTest, DISABLED_CodeGenTriGradient) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;
  UnstructuredIIRBuilder b;

  auto cell_f = b.field("cell_field", LocType::Cells);
  auto edge_f = b.field("edge_field", LocType::Edges);

  auto stencil_instantiation = b.build(
      "gradient",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Parallel,
          b.stage(LocType::Edges,
                  b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.stmt(b.assignExpr(
                                 b.at(edge_f),
                                 b.reduceOverNeighborExpr<float>(
                                     Op::plus, b.at(cell_f, HOffsetType::withOffset, 0), b.lit(0.),
                                     dawn::ast::LocationType::Edges, dawn::ast::LocationType::Cells,
                                     std::vector<float>({1., -1.})))))),
          b.stage(LocType::Cells,
                  b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.stmt(b.assignExpr(
                                 b.at(cell_f),
                                 b.reduceOverNeighborExpr<float>(
                                     Op::plus, b.at(edge_f, HOffsetType::withOffset, 0), b.lit(0.),
                                     dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges,
                                     std::vector<float>({1., 0., 0.})))))))));

  std::ofstream of("prototype/generated_triGradient.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dawn::CompilerUtil::dumpNaiveIco(of, stencil_instantiation);
}

TEST(CompilerTest, DISABLED_CodeGenQuadGradient) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;
  UnstructuredIIRBuilder b;

  auto cell_f = b.field("cell_field", LocType::Cells);
  auto edge_f = b.field("edge_field", LocType::Edges);

  auto stencil_instantiation = b.build(
      "gradient",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Parallel,
          b.stage(LocType::Edges,
                  b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.stmt(b.assignExpr(
                                 b.at(edge_f),
                                 b.reduceOverNeighborExpr<float>(
                                     Op::plus, b.at(cell_f, HOffsetType::withOffset, 0), b.lit(0.),
                                     dawn::ast::LocationType::Edges, dawn::ast::LocationType::Cells,
                                     std::vector<float>({1., -1.})))))),
          b.stage(LocType::Cells,
                  b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.stmt(b.assignExpr(
                                 b.at(cell_f),
                                 b.reduceOverNeighborExpr<float>(
                                     Op::plus, b.at(edge_f, HOffsetType::withOffset, 0), b.lit(0.),
                                     dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges,
                                     std::vector<float>({0.5, 0., 0., 0.5})))))))));

  std::ofstream of("prototype/generated_quadGradient.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dawn::CompilerUtil::dumpNaiveIco(of, stencil_instantiation);
}

TEST(CompilerTest, DISABLED_SparseDimension) {
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
                          b.lit(0.), dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges,
                          std::vector<float>({1., 1., 1., 1})))))))));

  std::ofstream of("prototype/generated_sparseDimension.hpp");
  dawn::CompilerUtil::dumpNaiveIco(of, stencil_instantiation);
}

TEST(CompilerTest, DISABLED_NestedReduce) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cell_f = b.field("cell_field", LocType::Cells);
  auto vertex_f = b.field("vertex_field", LocType::Vertices);

  // a nested reduction v->e->c
  auto stencil_instantiation = b.build(
      "nested",
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

  std::ofstream of("prototype/generated_NestedSimple.hpp");
  dawn::CompilerUtil::dumpNaiveIco(of, stencil_instantiation);
}

TEST(CompilerTest, DISABLED_NestedReduceField) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto cell_f = b.field("cell_field", LocType::Cells);
  auto edge_f = b.field("edge_field", LocType::Edges);
  auto vertex_f = b.field("vertex_field", LocType::Vertices);

  // a nested reduction v->e->c, the edge field is also consumed "along the way"
  auto stencil_instantiation = b.build(
      "nested",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Parallel,
          b.stage(LocType::Cells,
                  b.doMethod(
                      dawn::sir::Interval::Start, dawn::sir::Interval::End,
                      b.stmt(b.assignExpr(b.at(cell_f),
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

  std::ofstream of("prototype/generated_NestedWithField.hpp");
  dawn::CompilerUtil::dumpNaiveIco(of, stencil_instantiation);
}

TEST(CompilerTest, DISABLED_ICONStencil) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto vec = b.field("vec", LocType::Edges);
  auto div_vec = b.field("div_vec", LocType::Cells);
  auto rot_vec = b.field("rot_vec", LocType::Vertices);
  auto nabla2t1_vec = b.field("nabla2t1_vec", LocType::Edges);
  auto nabla2t2_vec = b.field("nabla2t2_vec", LocType::Edges);
  auto nabla2_vec = b.field("nabla2_vec", LocType::Edges);
  auto primal_edge_length = b.field("primal_edge_length", LocType::Edges);
  auto dual_edge_length = b.field("dual_edge_length", LocType::Edges);
  auto tangent_orientation = b.field("tangent_orientation", LocType::Edges);
  auto geofac_rot = b.field("geofac_rot", {LocType::Vertices, LocType::Edges});
  auto geofac_div = b.field("geofac_div", {LocType::Cells, LocType::Edges});

  // a nested reduction v->e->c, the edge field is also consumed "along the way"
  auto stencil_instantiation = b.build(
      "icon",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Parallel,
          b.stage(
              LocType::Vertices,
              b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                         b.stmt(b.assignExpr(
                             b.at(rot_vec),
                             b.reduceOverNeighborExpr(
                                 Op::plus, b.binaryExpr(b.at(vec), b.at(geofac_rot), Op::multiply),
                                 b.lit(0.), dawn::ast::LocationType::Vertices,
                                 dawn::ast::LocationType::Edges))))),
          b.stage(
              LocType::Cells,
              b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                         b.stmt(b.assignExpr(
                             b.at(div_vec),
                             b.reduceOverNeighborExpr(
                                 Op::plus, b.binaryExpr(b.at(vec), b.at(geofac_div), Op::multiply),
                                 b.lit(0.), dawn::ast::LocationType::Cells,
                                 dawn::ast::LocationType::Edges))))),
          b.stage(
              LocType::Edges,
              b.doMethod(
                  dawn::sir::Interval::Start, dawn::sir::Interval::End,
                  b.stmt(b.assignExpr(b.at(nabla2t1_vec),
                                      b.reduceOverNeighborExpr(Op::plus, b.at(rot_vec), b.lit(0.),
                                                               dawn::ast::LocationType::Edges,
                                                               dawn::ast::LocationType::Vertices,
                                                               std::vector<double>{-1, 1}))),
                  b.stmt(b.assignExpr(b.at(nabla2t1_vec),
                                      b.binaryExpr(b.binaryExpr(b.at(tangent_orientation),
                                                                b.at(nabla2t1_vec), Op::multiply),
                                                   b.at(primal_edge_length), Op::divide))),
                  b.stmt(b.assignExpr(b.at(nabla2t2_vec),
                                      b.reduceOverNeighborExpr(Op::plus, b.at(div_vec), b.lit(0.),
                                                               dawn::ast::LocationType::Edges,
                                                               dawn::ast::LocationType::Cells,
                                                               std::vector<double>{-1, 1}))),
                  b.stmt(b.assignExpr(
                      b.at(nabla2t2_vec),
                      b.binaryExpr(b.at(nabla2t2_vec), b.at(dual_edge_length), Op::divide))),
                  b.stmt(b.assignExpr(
                      b.at(nabla2_vec),
                      b.binaryExpr(b.at(nabla2t1_vec), b.at(nabla2t2_vec), Op::minus))))))));

  std::ofstream of("prototype/generated_iconLaplace.hpp");
  dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
  of.close();
}

} // anonymous namespace
