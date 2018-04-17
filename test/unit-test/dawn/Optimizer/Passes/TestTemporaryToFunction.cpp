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
#include "dawn/CodeGen/GridTools/GTCodeGen.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/SIR/SIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TemporaryToFunction : public ::testing::Test {

  dawn::DawnCompiler compiler_;

protected:
  TemporaryToFunction() {
    compiler_.getOptions().PassTmpToFunction = true;
    //    compiler_.getOptions().ReportPassTmpToFunction = true;
  }
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
    CG = make_unique<codegen::gt::GTCodeGen>(optimizer.get());
    auto translationUnit = CG->generateCode();

    if(optimizer->getDiagnostics().hasDiags()) {
      for(const auto& diag : optimizer->getDiagnostics().getQueue())
        std::cerr << "ERROR : " << diag->getMessage().c_str() << std::endl;
    }

    DAWN_ASSERT(translationUnit);
    //    for(auto pair : translationUnit->getStencils()) {
    //      std::cout << pair.first << " KKKKKKKKKKK " << pair.second << std::endl;
    //    }

    return optimizer->getStencilInstantiationMap()["compute_extent_test_stencil"]->getStencils();
  }
};

} // anonymous namespace
