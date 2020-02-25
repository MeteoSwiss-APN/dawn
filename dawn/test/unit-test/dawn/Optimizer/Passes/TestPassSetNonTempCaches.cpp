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
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>
#include <memory>

using namespace dawn;

namespace {

class TestPassSetNonTempCaches : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  virtual void SetUp() { options_.UseNonTempCaches = true; }

  void runTest(const std::string& filename, const std::vector<std::string>& cacheNames) {
    std::shared_ptr<iir::StencilInstantiation> instantiation =
        CompilerUtil::load(filename, options_, context_, TestEnvironment::path_);

    // Expect pass to succeed...
    PassSetNonTempCaches pass(*context_);
    ASSERT_TRUE(pass.run(instantiation));
    ASSERT_EQ(cacheNames, pass.getCachedFieldNames());
  }
};

// TEST_F(TestPassSetNonTempCaches, NoCache1) {
//   std::shared_ptr<iir::StencilInstantiation> instantiation =
//       IIRSerializer::deserialize("input/noCache.iir");
//   DiagnosticsEngine diag;
//   dawn::OptimizerContext::OptimizerContextOptions options_;
//   std::unique_ptr<OptimizerContext> context_ =
//       std::make_unique<OptimizerContext>(diag, options_, nullptr);
//   context_->getOptions().UseNonTempCaches = true;
//   PassSetNonTempCaches pass(*context_);
//   instantiation->dump();
//   pass.run(instantiation);
//   ASSERT_EQ(pass.getCachedFieldNames().size(), 0);
// }

// TEST_F(TestPassSetNonTempCaches, NoCache2) {
//   std::shared_ptr<iir::StencilInstantiation> instantiation =
//       IIRSerializer::deserialize("input/noCache2.iir");
//   DiagnosticsEngine diag;
//   dawn::OptimizerContext::OptimizerContextOptions options_;
//   std::unique_ptr<OptimizerContext> context_ =
//       std::make_unique<OptimizerContext>(diag, options_, nullptr);
//   context_->getOptions().UseNonTempCaches = true;
//   PassSetNonTempCaches pass(*context_);
//   instantiation->dump();
//   pass.run(instantiation);
//   ASSERT_EQ(pass.getCachedFieldNames().size(), 0);
// }

TEST_F(TestPassSetNonTempCaches, MultipleCaches1) {
  std::shared_ptr<iir::StencilInstantiation> instantiation =
      IIRSerializer::deserialize("input/field_aCached.iir");
  DiagnosticsEngine diag;
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_ =
      std::make_unique<OptimizerContext>(diag, options_, nullptr);
  context_->getOptions().SetNonTempCaches = true;
  PassSetNonTempCaches pass(*context_);
  for(int i = 0; i < 50; ++i)
    UIDGenerator::getInstance()->get();
  pass.run(instantiation);
  ASSERT_EQ(pass.getCachedFieldNames().size(), 1);
}

} // anonymous namespace
