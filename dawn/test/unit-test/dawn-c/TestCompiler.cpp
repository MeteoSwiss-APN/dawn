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
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/DiagnosticsEngine.h"
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

template <>
void dump<dawn::codegen::cuda::CudaCodeGen>(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
  dawn::DiagnosticsEngine diagnostics;
  using CG = dawn::codegen::cuda::CudaCodeGen;
  CG generator(ctx, diagnostics, 0, 0, 0, {0, 0, 0});
  auto tu = generator.generateCode();

  std::ostringstream ss;
  for(auto const& macroDefine : tu->getPPDefines())
    ss << macroDefine << "\n";

  ss << tu->getGlobals();
  for(auto const& s : tu->getStencils())
    ss << s.second;
  os << ss.str();
}

std::string read(const std::string& file) {
    std::ifstream is(file);
    std::string str((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    return str;
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
                  b.stage(b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                    b.block(b.stmt(b.assignExpr(b.at(out_f), b.at(in_f)))))))));
  std::ofstream of("/dev/null");
  dump<dawn::codegen::cxxnaive::CXXNaiveCodeGen>(of, stencil_instantiation);
}

TEST(CompilerTest, CompileGlobalIndexStencilNaive) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto in_f = b.field("in_field", FieldType::ijk);
  auto out_f = b.field("out_field", FieldType::ijk);

  auto stencil_instantiation = b.build(
      "generated", b.stencil(b.multistage(
                       LoopOrderKind::Parallel,
                       b.stage(b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                         b.block(b.stmt(b.assignExpr(b.at(out_f), b.at(in_f)))))),
                       b.stage(1, {0, 2},
                               b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                         b.block(b.stmt(b.assignExpr(b.at(out_f), b.lit(10)))))))));
  std::ofstream ofs("prototype/generated/global_indexing_naive.cpp");
  dump<dawn::codegen::cxxnaive::CXXNaiveCodeGen>(ofs, stencil_instantiation);

  std::string gen = read("prototype/generated/global_indexing_naive.cpp");
  std::string ref = read("prototype/reference/global_indexing_naive.cpp");
  ASSERT_EQ(gen, ref) << "Generated code does not match reference code";
}

TEST(CompilerTest, CompileGlobalIndexStencilCuda) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto in_f = b.field("in_field", FieldType::ijk);
  auto out_f = b.field("out_field", FieldType::ijk);

  auto stencil_instantiation = b.build(
      "generated", b.stencil(b.multistage(
                       LoopOrderKind::Parallel,
                       b.stage(b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                         b.block(b.stmt(b.assignExpr(b.at(out_f), b.at(in_f)))))),
                       b.stage(1, {0, 2},
                               b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                         b.block(b.stmt(b.assignExpr(b.at(out_f), b.lit(10)))))))));
  std::ofstream ofs("prototype/generated/global_indexing_cuda.cpp");
  dump<dawn::codegen::cuda::CudaCodeGen>(ofs, stencil_instantiation);

  std::string gen = read("prototype/generated/global_indexing_cuda.cpp");
  std::string ref = read("prototype/reference/global_indexing_cuda.cpp");
  ASSERT_EQ(gen, ref) << "Generated code does not match reference code";
}

TEST(CompilerTest, CompileLaplacian) {
    using namespace dawn::iir;
    using SInterval = dawn::sir::Interval;

//    // Define stencil code
//    std::ostringstream os;
//    os << "globals {\n"
//          "  double dx;\n"
//          "};\n"
//          "stencil laplacian {\n"
//          "storage out;\n"
//          "storage in;\n"
//          "Do() {\n"
//          "vertical_region(k_start, k_end) {\n"
//          "    out[i,j] = (-4 * in[i,j] + in[i+1,j] + in[i-1,j] +\n"
//          "        in[i,j-1] + in[i,j+1]) / (dx * dx);\n"
//          "}\n}\n};\n";
//    std::string code = os.str();
//    ASSERT_TRUE(!code.empty());
//
//    // Add header info and write to temporary file
//    std::string path = "/tmp/stencil.cpp";
//    std::ofstream ofs(path);
//    ASSERT_TRUE(ofs.is_open());
//    ofs << "#include \"gtclang_dsl_defs/gtclang_dsl.hpp\"\n";
//    ofs << "using namespace gtclang::dsl;\n\n";
//    ofs << code;
//    ofs.close();

  CartesianIIRBuilder b;
  auto in = b.field("in", FieldType::ijk);
  auto out = b.field("out", FieldType::ijk);
  auto dx = b.localvar("dx", dawn::BuiltinTypeID::Double);

  auto stencil_inst = b.build("generated",
    b.stencil(
      b.multistage(LoopOrderKind::Parallel,
        b.stage(
          b.vregion(SInterval::Start, SInterval::End, b.declareVar(dx),
            b.block(
              b.stmt(
                b.assignExpr(b.at(out),
                  b.binaryExpr(
                    b.binaryExpr(b.lit(-4),
                      b.binaryExpr(b.at(in),
                        b.binaryExpr(b.at(in, {1, 0, 0}),
                          b.binaryExpr(b.at(in, {-1, 0, 0}),
                            b.binaryExpr(b.at(in, {0, -1, 0}), b.at(in, {0, 1, 0}))
                    ) ) ), Op::multiply),
                    b.binaryExpr(b.at(dx), b.at(dx), Op::multiply), Op::divide)
            ) ) ) ) )
          ) ) );

  std::ofstream ofs("prototype/generated/laplacian_stencil.cpp");
  dump<dawn::codegen::cxxnaive::CXXNaiveCodeGen>(ofs, stencil_inst);
}

TEST(CompilerTest, CompileNonOverlapping) {
  using namespace dawn::iir;
  using SInterval = dawn::sir::Interval;

  CartesianIIRBuilder b;
  auto in = b.field("in", FieldType::ijk);
  auto out = b.field("out", FieldType::ijk);
  auto dx = b.localvar("dx", dawn::BuiltinTypeID::Double);

  auto stencil_inst = b.build("generated",
    b.stencil(
      b.multistage(LoopOrderKind::Parallel,
        b.stage(
          b.vregion(SInterval(SInterval::Start, 10), b.declareVar(dx),
            b.block(
              b.stmt(
                b.assignExpr(b.at(out),
                  b.binaryExpr(
                    b.binaryExpr(b.lit(-4),
                      b.binaryExpr(b.at(in),
                        b.binaryExpr(b.at(in, {1, 0, 0}),
                          b.binaryExpr(b.at(in, {-1, 0, 0}),
                            b.binaryExpr(b.at(in, {0, -1, 0}), b.at(in, {0, 1, 0}))
                    ) ) ), Op::multiply),
                    b.binaryExpr(b.at(dx), b.at(dx), Op::multiply), Op::divide)
            ) ) ) ) )
         , b.stage(b.vregion(SInterval(15, SInterval::End),
            b.block(
              b.stmt(
                b.assignExpr(b.at(out), b.lit(10))
  ) ) ) ) ) ) );


  std::ofstream ofs("prototype/generated/nonoverlapping_stencil.cpp");
  dump<dawn::codegen::cxxnaive::CXXNaiveCodeGen>(ofs, stencil_inst);
//
//    std::string gen = read("prototype/generated/global_indexing_naive.cpp");
//    std::string ref = read("prototype/reference/global_indexing_naive.cpp");
//    ASSERT_EQ(gen, ref) << "Generated code does not match reference code";
}

TEST(CompilerTest, DISABLED_CodeGenSumEdgeToCells) {
  using namespace dawn::iir;
  using LocType = dawn::ast::Expr::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Edges);
  auto out_f = b.field("out_field", LocType::Cells);
  auto cnt = b.localvar("cnt", dawn::BuiltinTypeID::Integer);

  auto stencil_instantiation = b.build(
      "generated",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Edges, b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                            b.stmt(b.assignExpr(b.at(in_f), b.lit(10))))),
          b.stage(b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                            b.stmt(b.assignExpr(
                                b.at(out_f), b.reduceOverNeighborExpr(
                                                 Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                                 b.lit(0.), LocType::Cells, LocType::Edges))))))));

  std::ofstream of("prototype/generated_copyEdgeToCell.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
  of.close();
}

TEST(CompilerTest, DISABLED_SumVertical) {
  using namespace dawn::iir;
  using LocType = dawn::ast::Expr::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Cells);
  auto out_f = b.field("out_field", LocType::Cells);

  auto stencil_instantiation = b.build(
      "generated",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Cells,
                  b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End, 1, -1,
                            b.stmt(b.assignExpr(b.at(out_f),
                                                b.binaryExpr(b.at(in_f, HOffsetType::noOffset, +1),
                                                             b.at(in_f, HOffsetType::noOffset, -1),
                                                             Op::plus))))))));

  std::ofstream of("prototype/generated_verticalSum.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
  of.close();
}

//TEST(CompilerTest, DISABLED_CodeGenDiffusion) {
TEST(CompilerTest, CodeGenDiffusion) {
  using namespace dawn::iir;
  using LocType = dawn::ast::Expr::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Cells);
  auto out_f = b.field("out_field", LocType::Cells);
  auto cnt = b.localvar("cnt", dawn::BuiltinTypeID::Integer);

  auto stencil_instantiation = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Parallel,
          b.stage(b.vregion(
              dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(cnt),
              b.stmt(b.assignExpr(b.at(cnt),
                                  b.reduceOverNeighborExpr(Op::plus, b.lit(1), b.lit(0),
                                                           dawn::ast::Expr::LocationType::Cells,
                                                           dawn::ast::Expr::LocationType::Cells))),
              b.stmt(b.assignExpr(
                  b.at(out_f),
                  b.reduceOverNeighborExpr(
                      Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                      b.binaryExpr(b.unaryExpr(b.at(cnt), Op::minus),
                                   b.at(in_f, HOffsetType::withOffset, 0), Op::multiply),
                      dawn::ast::Expr::LocationType::Cells, dawn::ast::Expr::LocationType::Cells))),
              b.stmt(b.assignExpr(b.at(out_f),
                                  b.binaryExpr(b.at(in_f),
                                               b.binaryExpr(b.lit(0.1), b.at(out_f), Op::multiply),
                                               Op::plus))))))));

  std::ofstream of("prototype/generated_diffusion.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dump<dawn::codegen::cxxnaive::CXXNaiveCodeGen>(of, stencil_instantiation);
  of.close();
}

TEST(CompilerTest, DISABLED_CodeGenTriGradient) {
  using namespace dawn::iir;
  using LocType = dawn::ast::Expr::LocationType;
  UnstructuredIIRBuilder b;

  auto cell_f = b.field("cell_field", LocType::Cells);
  auto edge_f = b.field("edge_field", LocType::Edges);

  auto stencil_instantiation = b.build(
      "gradient",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Parallel,
          b.stage(
              LocType::Edges,
              b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                        b.stmt(b.assignExpr(b.at(edge_f),
                                            b.reduceOverNeighborExpr<float>(
                                                Op::plus, b.at(cell_f, HOffsetType::withOffset, 0),
                                                b.lit(0.), dawn::ast::Expr::LocationType::Edges,
                                                dawn::ast::Expr::LocationType::Cells,
                                                std::vector<float>({1., -1.})))))),
          b.stage(
              LocType::Cells,
              b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                        b.stmt(b.assignExpr(b.at(cell_f),
                                            b.reduceOverNeighborExpr<float>(
                                                Op::plus, b.at(edge_f, HOffsetType::withOffset, 0),
                                                b.lit(0.), dawn::ast::Expr::LocationType::Cells,
                                                dawn::ast::Expr::LocationType::Edges,
                                                std::vector<float>({1., 0., 0.})))))))));

  std::ofstream of("prototype/generated_triGradient.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
  of.close();
}

TEST(CompilerTest, DISABLED_CodeGenQuadGradient) {
  using namespace dawn::iir;
  using LocType = dawn::ast::Expr::LocationType;
  UnstructuredIIRBuilder b;

  auto cell_f = b.field("cell_field", LocType::Cells);
  auto edge_f = b.field("edge_field", LocType::Edges);

  auto stencil_instantiation = b.build(
      "gradient",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Parallel,
          b.stage(
              LocType::Edges,
              b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                        b.stmt(b.assignExpr(b.at(edge_f),
                                            b.reduceOverNeighborExpr<float>(
                                                Op::plus, b.at(cell_f, HOffsetType::withOffset, 0),
                                                b.lit(0.), dawn::ast::Expr::LocationType::Edges,
                                                dawn::ast::Expr::LocationType::Cells,
                                                std::vector<float>({1., -1.})))))),
          b.stage(
              LocType::Cells,
              b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                        b.stmt(b.assignExpr(b.at(cell_f),
                                            b.reduceOverNeighborExpr<float>(
                                                Op::plus, b.at(edge_f, HOffsetType::withOffset, 0),
                                                b.lit(0.), dawn::ast::Expr::LocationType::Cells,
                                                dawn::ast::Expr::LocationType::Edges,
                                                std::vector<float>({0.5, 0., 0., 0.5})))))))));

  std::ofstream of("prototype/generated_quadGradient.hpp");
  DAWN_ASSERT_MSG(of, "file could not be opened. Binary must be called from dawn/dawn");
  dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
  of.close();
}

} // anonymous namespace
