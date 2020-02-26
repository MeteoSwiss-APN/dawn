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

#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassSetNonTempCaches.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include <gtest/gtest.h>
#include <memory>

using namespace dawn;

namespace {

class TestPassSetNonTempCaches : public ::testing::Test {
protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;
  DiagnosticsEngine diag;

  virtual void SetUp() {
    context_ = std::make_unique<OptimizerContext>(diag, options_, nullptr);
    context_->getOptions().UseNonTempCaches = true;
  }
};

TEST_F(TestPassSetNonTempCaches, NoCache1) {
  std::shared_ptr<iir::StencilInstantiation> instantiation =
      IIRSerializer::deserialize("input/TestNonTempCache_01.iir");
  PassSetNonTempCaches pass(*context_);
  pass.run(instantiation);
  ASSERT_EQ(pass.getCachedFieldNames().size(), 0);
}

TEST_F(TestPassSetNonTempCaches, NoCache2) {
  std::shared_ptr<iir::StencilInstantiation> instantiation =
      IIRSerializer::deserialize("input/TestNonTempCache_02.iir");
  // DiagnosticsEngine diag;
  // dawn::OptimizerContext::OptimizerContextOptions options_;
  // std::unique_ptr<OptimizerContext> context_ =
  //     std::make_unique<OptimizerContext>(diag, options_, nullptr);
  // context_->getOptions().UseNonTempCaches = true;
  PassSetNonTempCaches pass(*context_);
  pass.run(instantiation);
  ASSERT_EQ(pass.getCachedFieldNames().size(), 0);
}

TEST_F(TestPassSetNonTempCaches, MultipleCaches1) {
  std::shared_ptr<iir::StencilInstantiation> instantiation =
      IIRSerializer::deserialize("input/TestNonTempCache_03.iir");
  // DiagnosticsEngine diag;
  // dawn::OptimizerContext::OptimizerContextOptions options_;
  // std::unique_ptr<OptimizerContext> context_ =
  //     std::make_unique<OptimizerContext>(diag, options_, nullptr);
  // context_->getOptions().SetNonTempCaches = true;
  PassSetNonTempCaches pass(*context_);
  pass.run(instantiation);
  ASSERT_EQ(pass.getCachedFieldNames().size(), 1);
}

} // anonymous namespace
