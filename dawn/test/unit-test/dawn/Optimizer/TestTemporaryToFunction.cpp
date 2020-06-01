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

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/LoopOrder.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassTemporaryToStencilFunction.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TestPassTemporaryToFunction : public ::testing::Test {
public:
  TestPassTemporaryToFunction() {
    options_.TmpToStencilFunction = true;
    context_ = std::make_unique<OptimizerContext>(options_, nullptr);
  }

protected:
  OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  std::shared_ptr<iir::StencilInstantiation> runPass(const std::string& filename) {
    auto instantiation = IIRSerializer::deserialize(filename);
    EXPECT_TRUE(instantiation->getIIR()->getChildren().size() == 1);
    for(auto& stmt : iterateIIROverStmt(*instantiation->getIIR())) {
      stmt->getData<iir::IIRStmtData>().StackTrace = std::vector<ast::StencilCall*>();
    }

    // run temp to function passs
    PassTemporaryToStencilFunction tmpToFun(*context_);
    tmpToFun.run(instantiation);

    return instantiation;
  }
};

TEST_F(TestPassTemporaryToFunction, TmpToFunDo) {
  /*
    storage in, out;
    var tmp;
    void Do() {
      vertical_region(k_start, k_end) {
        tmp = in + 1; //to fun here
        out = tmp[i - 1] + 2;
      }
    }
  */
  auto instantiation = runPass("input/TestTmpToFunDo.iir");
  const auto& funs = instantiation->getIIR()->getStencilFunctions();
  // we expect that one stencil function has been generated
  ASSERT_TRUE(funs.size() == 1);
}

TEST_F(TestPassTemporaryToFunction, TmpToFunDont) {
  /*
    storage a, b, out;
    var c;
    Do {
      vertical_region(k_start, k_end) {
        c = a + b;  //does not qualify to be "promoted" function
        out = c;
      }
    }
  */
  auto instantiation = runPass("input/TestTmpToFunDont.iir");
  // we expect that no stencil function has been generated
  const auto& funs = instantiation->getIIR()->getStencilFunctions();
  ASSERT_TRUE(funs.size() == 0);
}

} // anonymous namespace
