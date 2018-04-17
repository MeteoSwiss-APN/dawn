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

#include "dawn/Support/Assert.h"
#include "dawn/Support/STLExtras.h"
#include "dawn/Unittest/UnittestLogger.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"
#include <gtest/gtest.h>

std::string TestEnvironment::path_ = "";

int main(int argc, char* argv[]) {

  // Initialize gtest
  testing::InitGoogleTest(&argc, argv);

  DAWN_ASSERT_MSG((argc == 2), "wrong number of arguments");

  std::string path = argv[1];

  TestEnvironment::path_ = path;
  ::testing::AddGlobalTestEnvironment(new TestEnvironment());
  return RUN_ALL_TESTS();
}
