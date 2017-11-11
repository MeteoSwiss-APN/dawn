//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gtclang/Unittest/UnittestEnvironment.h"
#include <gtest/gtest.h>

using namespace gtclang;

namespace {

TEST(UnittestEnvironmentTest, TestCaseName) {
  auto& env = UnittestEnvironment::getSingleton();
  EXPECT_STREQ(env.testCaseName().c_str(), "UnittestEnvironmentTest");
}

TEST(UnittestEnvironmentTest, TestName) {
  auto& env = UnittestEnvironment::getSingleton();
  EXPECT_STREQ(env.testName().c_str(), "TestName");
}

} // anonymous namespace
