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
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"

#include "dawn/Support/DiagnosticsEngine.h"

#include <fstream>
#include <gtest/gtest.h>
#include <memory>

using namespace dawn;

namespace {

class TestPassFieldVersioning : public ::testing::Test {
public:
  TestPassFieldVersioning() {
    context_ = std::make_unique<OptimizerContext>(diagnostics_, options_, nullptr);
  }

protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  DiagnosticsEngine diagnostics_;
  std::unique_ptr<OptimizerContext> context_;

  void raceConditionTest(const std::string& filename) {
    context_->getDiagnostics().clear();
    std::shared_ptr<iir::StencilInstantiation> instantiation = IIRSerializer::deserialize(filename);

    // Expect pass to fail...
    dawn::PassFieldVersioning pass(*context_);
    ASSERT_FALSE(pass.run(instantiation));
    ASSERT_TRUE(context_->getDiagnostics().hasErrors());
  }

  void versioningTest(const std::string& filename) {
    context_->getDiagnostics().clear();
    std::shared_ptr<iir::StencilInstantiation> instantiation = IIRSerializer::deserialize(filename);

    // Expect pass to succeed...
    dawn::PassFieldVersioning pass(*context_);
    ASSERT_TRUE(pass.run(instantiation));
  }
};

TEST_F(TestPassFieldVersioning, RaceCondition1) {
  /*
  vertical_region(k_start, k_end) {
    if(field_a > 0.0) {
      field_b = field_a;
      field_a = field_b(i + 1);
    }
  }
  */
  raceConditionTest("input/TestPassFieldVersioning_01.iir");
}

TEST_F(TestPassFieldVersioning, RaceCondition2) {
  /*
  vertical_region(k_start, k_end) {
    if(field_a > 0.0) {
      field_b = field_a;
      double b = field_b(i + 1);
      field_a = b;
    }
  }
  */
  raceConditionTest("input/TestPassFieldVersioning_02.iir");
}

TEST_F(TestPassFieldVersioning, RaceCondition3) {
  /*
  stencil_function TestFunction {
    storage field_a;

    Do { return field_a(i + 1); }
  };
  vertical_region(k_start, k_end) {
    field_a = TestFunction(field_a);
  }
  Note: Inlined
  */
  raceConditionTest("input/TestPassFieldVersioning_03.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest1) {
  /*
  vertical_region(k_start, k_end) { field_a = field_b; }
  */
  versioningTest("input/TestPassFieldVersioning_04.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest2) {
  /*
  vertical_region(k_start, k_end) {
    field_a = field_a(i + 1);
  }
  */
  versioningTest("input/TestPassFieldVersioning_05.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest3) {
  /*
  vertical_region(k_start, k_end) {
    field_b = field_a(i + 1);
    field_a = field_b;
  }
  */
  versioningTest("input/TestPassFieldVersioning_06.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest4) {
  /*
  vertical_region(k_start, k_end) {
    tmp = field_a(i + 1) + field_b(i + 1);
    field_a = tmp;
    field_b = tmp;
  }
  */
  versioningTest("input/TestPassFieldVersioning_07.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest5) {
  /*
  vertical_region(k_start, k_end) {
    tmp1 = field_a(i + 1);
    tmp2 = tmp1;
    field_a = tmp2;
  }
  */
  versioningTest("input/TestPassFieldVersioning_08.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest6) {
  /*
  vertical_region(k_start, k_end) {
      tmp = field(i + 1);
      field = tmp;

      tmp = field(i + 1);
      field = tmp;
  }
  */
  versioningTest("input/TestPassFieldVersioning_09.iir");
}

TEST_F(TestPassFieldVersioning, VersioningTest7) {
  /*
  stencil_function TestFunction {
  storage field_a, field_b;

  Do {
      field_b = field_a;
      field_a = field_b(i + 1);
      return 0.0;
    }
  };
  vertical_region(k_start, k_end) {
        TestFunction(field_a, field_b);
      }
    Note: Inlined
*/
  versioningTest("input/TestPassFieldVersioning_10.iir");
}
} // anonymous namespace
