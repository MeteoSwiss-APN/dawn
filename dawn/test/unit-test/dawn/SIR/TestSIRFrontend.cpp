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

#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassSetNonTempCaches.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Unittest/CompilerUtil.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestSIRFrontend : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  void runTest(const std::string& irFilename, const std::string& srcFilename = "") {
    std::shared_ptr<iir::StencilInstantiation> instantiation =
        CompilerUtil::load(irFilename, options_, context_);

    // Run all passes
    ASSERT_TRUE(CompilerUtil::runPasses(context_, instantiation));

    // Code gen...
    ASSERT_GT(CompilerUtil::generate(instantiation, srcFilename).size(), 0);
  }
};

TEST_F(TestSIRFrontend, TridiagonalSolveGTClang) {
  runTest("input/tridiagonal_solve_gtclang.sir", "output/tridiagonal_solve_gtclang.cpp");
}

TEST_F(TestSIRFrontend, TridiagonalSolveGT4Py) {
  runTest("input/tridiagonal_solve_gt4py.sir", "output/tridiagonal_solve_gt4py.cpp");
}

TEST_F(TestSIRFrontend, TridiagonalSolveGT4PyCuda) {
  runTest("input/tridiagonal_solve_gt4py.sir", "output/tridiagonal_solve_gt4py.cu");
}

TEST_F(TestSIRFrontend, HorizontalDifferenceGTClang) {
  runTest("input/horizontal_difference_gtclang.sir", "output/horizontal_difference_gtclang.cpp");
}

TEST_F(TestSIRFrontend, HorizontalDifferenceGT4Py) {
  runTest("input/horizontal_difference_gt4py.sir", "output/horizontal_difference_gt4py.cpp");
}

TEST_F(TestSIRFrontend, HorizontalDifferenceGT4PyCuda) {
  runTest("input/horizontal_difference_gt4py.sir", "output/horizontal_difference_gt4py.cu");
}

TEST_F(TestSIRFrontend, BurgersDemoGTClang) {
  runTest("input/burgers_demo_gtclang.sir", "output/burgers_demo_gtclang.cpp");
}

TEST_F(TestSIRFrontend, BurgersDemoGT4Py) {
  runTest("input/burgers_demo_gt4py.sir", "output/burgers_demo_gt4py.cpp");
}

TEST_F(TestSIRFrontend, BurgersDemoGT4PyCuda) {
  runTest("input/burgers_demo_gt4py.sir", "output/burgers_demo_gt4py.cu");
}

TEST_F(TestSIRFrontend, CoriolisGT4Py) {
  runTest("input/coriolis_stencil_gt4py.sir", "output/coriolis_stencil_gt4py.cpp");
}

TEST_F(TestSIRFrontend, CoriolisGT4PyCuda) {
  runTest("input/coriolis_stencil_gt4py.sir", "output/coriolis_stencil_gt4py.cu");
}

TEST_F(TestSIRFrontend, YPPMMainAlGT4Py) {
  runTest("input/yppm_main_al_gt4py.sir", "output/yppm_main_al_gt4py.cpp");
}

TEST_F(TestSIRFrontend, YPPMAlXEdgeGT4Py) {
  runTest("input/yppm_al_x_edge_gt4py.sir", "output/yppm_al_x_edge_gt4py.cpp");
}

} // anonymous namespace
