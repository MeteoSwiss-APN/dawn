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
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <gtest/gtest.h>
#include <memory>

using namespace dawn;

namespace {

class TestPassFieldVersioning : public ::testing::Test {
public:
  TestPassFieldVersioning() { context_ = std::make_unique<OptimizerContext>(options_, nullptr); }

protected:
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<OptimizerContext> context_;

  void raceConditionTest(const std::string& filename) {
    auto instantiation = IIRSerializer::deserialize(filename);

    // Expect pass to fail...
    dawn::PassFieldVersioning pass(*context_);
    EXPECT_ANY_THROW(pass.run(instantiation));
  }

  std::shared_ptr<iir::StencilInstantiation> versioningTest(const std::string& filename) {
    auto instantiation = IIRSerializer::deserialize(filename);

    // Expect pass to succeed...
    dawn::PassFieldVersioning pass(*context_);
    pass.run(instantiation);
    return instantiation;
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
  auto instantiation = versioningTest("input/TestPassFieldVersioning_04.iir");
  int idA = instantiation->getMetaData().getAccessIDFromName("field_a");
  ASSERT_FALSE(instantiation->getMetaData().isMultiVersionedField(idA));

  int idB = instantiation->getMetaData().getAccessIDFromName("field_b");
  ASSERT_FALSE(instantiation->getMetaData().isMultiVersionedField(idB));
}

TEST_F(TestPassFieldVersioning, VersioningTest2) {
  /*
  vertical_region(k_start, k_end) {
    field_a = field_a(i + 1);
  }
  */
  auto instantiation = versioningTest("input/TestPassFieldVersioning_05.iir");
  int idA = instantiation->getMetaData().getAccessIDFromName("field_a");
  ASSERT_TRUE(instantiation->getMetaData().isMultiVersionedField(idA));
}

TEST_F(TestPassFieldVersioning, VersioningTest3) {
  /*
  vertical_region(k_start, k_end) {
    field_b = field_a(i + 1);
    field_a = field_b;
  }
  */
  auto instantiation = versioningTest("input/TestPassFieldVersioning_06.iir");

  int idA = instantiation->getMetaData().getAccessIDFromName("field_a");
  ASSERT_TRUE(instantiation->getMetaData().isMultiVersionedField(idA));

  int idB = instantiation->getMetaData().getAccessIDFromName("field_b");
  ASSERT_FALSE(instantiation->getMetaData().isMultiVersionedField(idB));
}

TEST_F(TestPassFieldVersioning, VersioningTest4) {
  /*
  vertical_region(k_start, k_end) {
    tmp = field_a(i + 1) + field_b(i + 1);
    field_a = tmp;
    field_b = tmp;
  }
  */
  auto instantiation = versioningTest("input/TestPassFieldVersioning_07.iir");

  int idA = instantiation->getMetaData().getAccessIDFromName("field_a");
  ASSERT_TRUE(instantiation->getMetaData().isMultiVersionedField(idA));

  int idB = instantiation->getMetaData().getAccessIDFromName("field_b");
  ASSERT_TRUE(instantiation->getMetaData().isMultiVersionedField(idB));
}

TEST_F(TestPassFieldVersioning, VersioningTest5) {
  /*
  vertical_region(k_start, k_end) {
    tmp1 = field_a(i + 1);
    tmp2 = tmp1;
    field_a = tmp2;
  }
  */
  auto instantiation = versioningTest("input/TestPassFieldVersioning_08.iir");

  int idA = instantiation->getMetaData().getAccessIDFromName("field_a");
  ASSERT_TRUE(instantiation->getMetaData().isMultiVersionedField(idA));

  int idTmp1 = instantiation->getMetaData().getAccessIDFromName("tmp1");
  ASSERT_FALSE(instantiation->getMetaData().isMultiVersionedField(idTmp1));

  int idTmp2 = instantiation->getMetaData().getAccessIDFromName("tmp2");
  ASSERT_FALSE(instantiation->getMetaData().isMultiVersionedField(idTmp2));
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
  auto instantiation = versioningTest("input/TestPassFieldVersioning_09.iir");
  int idField = instantiation->getMetaData().getAccessIDFromName("field");
  ASSERT_TRUE(instantiation->getMetaData().isMultiVersionedField(idField));
  auto versions = instantiation->getMetaData().getVersionsOf(idField);
  ASSERT_EQ(versions->size(), 2);

  int idTmp = instantiation->getMetaData().getAccessIDFromName("tmp");
  ASSERT_TRUE(instantiation->getMetaData().isMultiVersionedField(idTmp));
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
  auto instantiation = versioningTest("input/TestPassFieldVersioning_10.iir");
  int idA = instantiation->getMetaData().getAccessIDFromName("field_a");
  ASSERT_TRUE(instantiation->getMetaData().isMultiVersionedField(idA));
}
} // anonymous namespace
