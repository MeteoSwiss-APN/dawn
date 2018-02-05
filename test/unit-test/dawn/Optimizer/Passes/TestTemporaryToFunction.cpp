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

#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/GridTools/GTCodeGen.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIRSerializer.h"
#include "dawn/Support/EditDistance.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringSwitch.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include "test/unit-test/dawn/Optimizer/Passes/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>
#include <streambuf>

#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Optimizer/PassDataLocalityMetric.h"
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/Optimizer/PassMultiStageSplitter.h"
#include "dawn/Optimizer/PassPrintStencilGraph.h"
#include "dawn/Optimizer/PassSSA.h"
#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/Optimizer/PassSetNonTempCaches.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassSetStageName.h"
#include "dawn/Optimizer/PassStageMerger.h"
#include "dawn/Optimizer/PassStageReordering.h"
#include "dawn/Optimizer/PassStageSplitter.h"
#include "dawn/Optimizer/PassStencilSplitter.h"
#include "dawn/Optimizer/PassTemporaryFirstAccess.h"
#include "dawn/Optimizer/PassTemporaryMerger.h"
#include "dawn/Optimizer/PassTemporaryToStencilFunction.h"
#include "dawn/Optimizer/PassTemporaryType.h"

using namespace dawn;

namespace {

class TemporaryToFunction : public ::testing::Test {
  std::unique_ptr<dawn::Options> compileOptions_;

  dawn::DawnCompiler compiler_;

protected:
  TemporaryToFunction() : compiler_(compileOptions_.get()) {}
  virtual void SetUp() {}

  std::vector<std::shared_ptr<Stencil>> loadTest(std::string sirFilename) {

    std::string filename = TestEnvironment::path_ + "/" + sirFilename;
    std::ifstream file(filename);
    DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

    std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::shared_ptr<SIR> sir =
        SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::SK_Json);

    std::unique_ptr<OptimizerContext> optimizer = compiler_.runOptimizer(sir);
    // Report diganostics
    if(compiler_.getDiagnostics().hasDiags()) {
      for(const auto& diag : compiler_.getDiagnostics().getQueue())
        std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
      throw std::runtime_error("compilation failed");
    }

    DAWN_ASSERT_MSG((optimizer->getStencilInstantiationMap().count("compute_extent_test_stencil")),
                    "compute_extent_test_stencil not found in sir");

    // Generate code
    std::unique_ptr<codegen::CodeGen> CG;
    //    switch(codeGen) {
    //    case CodeGenKind::CG_GTClang:
    CG = make_unique<codegen::gt::GTCodeGen>(optimizer.get());
    //      break;
    //    case CodeGenKind::CG_GTClangNaiveCXX:
    //      CG = make_unique<codegen::cxxnaive::CXXNaiveCodeGen>(optimizer.get());
    //      break;
    //    case CodeGenKind::CG_GTClangOptCXX:
    //      dawn_unreachable("GTClangOptCXX not supported yet");
    //      break;
    //    }
    auto gg = CG->generateCode();

    if(optimizer->getDiagnostics().hasDiags()) {
      for(const auto& diag : optimizer->getDiagnostics().getQueue())
        std::cerr << "ERROR : " << diag->getMessage().c_str() << std::endl;
    }

    DAWN_ASSERT(gg);
    for(auto pair : gg->getStencils()) {
      std::cout << pair.first << " KKKKKKKKKKK " << pair.second << std::endl;
    }

    return optimizer->getStencilInstantiationMap()["compute_extent_test_stencil"]->getStencils();
  }
};

TEST_F(TemporaryToFunction, test_stencil_03) {
  auto stencils = loadTest("compute_extent_test_stencil_03.sir");
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-1, 1, -1, 2, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{-1, 0, -1, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(2)->getExtents() == Extents{0, 0, 0, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(3)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));
}

} // anonymous namespace
