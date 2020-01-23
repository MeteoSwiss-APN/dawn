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

#include "gtclang/Unittest/GTClang.h"
#include "gtclang/Unittest/UnittestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>

using namespace gtclang;

namespace {

TEST(PreprocessingTest, Include) {
  auto flags = UnittestEnvironment::getSingleton().getFlagManager().getDefaultFlags();
  std::string filename = "test_stencil_w_include.cpp";
  auto pair1 = GTClang::run({filename, "-fno-codegen"}, flags);
  ASSERT_TRUE(pair1.first);

  filename = "test_stencil_no_include.cpp";
  auto pair2 = GTClang::run({filename, "-fno-codegen"}, flags);
  ASSERT_TRUE(pair2.first);
}

} // anonymous namespace
