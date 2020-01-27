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
#include "dawn/Optimizer/PassIntervalPartitioner.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/FileUtil.h"
//#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

using stencilInstantiationContext =
    std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>;

class TestIntegrityChecker : public ::testing::Test {
  std::unique_ptr<OptimizerContext> context_;

protected:
  virtual void SetUp() {
    dawn::DiagnosticsEngine diag;
    std::shared_ptr<SIR> sir = std::make_shared<SIR>(ast::GridType::Cartesian);
    dawn::OptimizerContext::OptimizerContextOptions options;
    //options.PartitionIntervals = true;
    context_ = std::make_unique<OptimizerContext>(diag, options, sir);
  }

  const std::unique_ptr<OptimizerContext>& getContext() { return context_; }

  stencilInstantiationContext compile(std::shared_ptr<SIR> sir) {
    std::unique_ptr<dawn::Options> options;
    DawnCompiler compiler(options.get());
    auto optimizer = compiler.runOptimizer(sir);

    if(compiler.getDiagnostics().hasDiags()) {
      for(const auto& diag : compiler.getDiagnostics().getQueue()) {
        std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
      }
      throw std::runtime_error("Compilation failed");
    }

    return optimizer->getStencilInstantiationMap();
  }
};

TEST_F(TestIntegrityChecker, GlobalsOptimizedAway) {
  std::string json = dawn::readFile("input/globals_opt_away.sir");
  std::shared_ptr<SIR> sir =
      SIRSerializer::deserializeFromString(json, SIRSerializer::Format::Json);

  try {
    compile(sir);
    FAIL() << "Semantic error not thrown";
  } catch(SemanticError& error) {
    SUCCEED();
  }
}

} // anonymous namespace
