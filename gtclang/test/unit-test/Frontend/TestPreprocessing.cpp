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

#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/UIDGenerator.h"
#include "gtclang/Unittest/GTClang.h"
#include "gtclang/Unittest/UnittestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace gtclang;

namespace {

TEST(PreprocessingTest, Include) {
  auto flags = UnittestEnvironment::getSingleton().getFlagManager().getDefaultFlags();

  std::string sirString1, sirString2;
  {
    const std::string filename = "input/test_stencil_w_include.cpp";
    dawn::UIDGenerator::getInstance()->reset();
    auto [passed, sir] = GTClang::run({filename, "-fno-codegen"}, flags);
    ASSERT_TRUE(passed);
    ASSERT_TRUE(!sir->Stencils.empty());
    sir->Filename = "";
    sirString1 = dawn::SIRSerializer::serializeToString(sir);
  }

  {
    const std::string filename = "input/test_stencil_no_include.cpp";
    dawn::UIDGenerator::getInstance()->reset();
    auto [passed, sir] = GTClang::run({filename, "-fno-codegen"}, flags);
    ASSERT_TRUE(passed);
    ASSERT_TRUE(!sir->Stencils.empty());
    sir->Filename = "";
    sirString2 = dawn::SIRSerializer::serializeToString(sir);
  }

  ASSERT_EQ(sirString1, sirString2);
}

} // anonymous namespace
