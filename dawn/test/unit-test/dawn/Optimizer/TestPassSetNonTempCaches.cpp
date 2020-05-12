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

#include <gtest/gtest.h>
#include <memory>

using namespace dawn;

namespace {

TEST(TestPassSetNonTempCaches, NoCache1) {
  /*
    vertical_region(k_start + 1, k_end - 1) { field_b = field_a[i - 1]; }
  */
  std::shared_ptr<iir::StencilInstantiation> instantiation =
      IIRSerializer::deserialize("input/TestNonTempCache_01.iir");
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_ =
      std::make_unique<OptimizerContext>(options_, nullptr);
  context_->getOptions().UseNonTempCaches = true;
  PassSetNonTempCaches pass(*context_);
  pass.run(instantiation);
  ASSERT_EQ(pass.getCachedFieldNames().size(), 0);
}

TEST(TestPassSetNonTempCaches, NoCache2) {
  /*
    vertical_region(k_start + 1, k_end - 1) {
      field_a = 10;
      field_b = field_a[i - 1];
      field_c = field_a[k - 1];
    }
  */
  std::shared_ptr<iir::StencilInstantiation> instantiation =
      IIRSerializer::deserialize("input/TestNonTempCache_02.iir");
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_ =
      std::make_unique<OptimizerContext>(options_, nullptr);
  context_->getOptions().UseNonTempCaches = true;
  PassSetNonTempCaches pass(*context_);
  pass.run(instantiation);
  ASSERT_EQ(pass.getCachedFieldNames().size(), 0);
}

TEST(TestPassSetNonTempCaches, MultipleCaches1) {
  /*
    vertical_region(k_start + 1, k_end - 1) {
      field_a = 10;
      field_b = field_a[i - 1];
    }
  */
  std::shared_ptr<iir::StencilInstantiation> instantiation =
      IIRSerializer::deserialize("input/TestNonTempCache_03.iir");
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_ =
      std::make_unique<OptimizerContext>(options_, nullptr);
  context_->getOptions().SetNonTempCaches = true;
  PassSetNonTempCaches pass(*context_);
  pass.run(instantiation);
  ASSERT_EQ(pass.getCachedFieldNames().size(), 1);
}

} // anonymous namespace
