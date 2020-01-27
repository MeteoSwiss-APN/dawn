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

  std::string filename = "input/test_stencil_w_include.cpp";
  dawn::UIDGenerator::getInstance()->reset();
  auto pair1 = GTClang::run({filename, "-fno-codegen"}, flags);
  ASSERT_TRUE(pair1.first);
  dawn::SIR* sir1 = pair1.second.get();
  ASSERT_TRUE(sir1->Stencils.size() > 0);

  filename = "input/test_stencil_no_include.cpp";
  dawn::UIDGenerator::getInstance()->reset();
  auto pair2 = GTClang::run({filename, "-fno-codegen"}, flags);
  ASSERT_TRUE(pair2.first);
  dawn::SIR* sir2 = pair2.second.get();
  ASSERT_TRUE(sir2->Stencils.size() > 0);

  sir2->Filename = sir1->Filename;
  std::string sirStr1 = dawn::SIRSerializer::serializeToString(sir1);
  std::string sirStr2 = dawn::SIRSerializer::serializeToString(sir2);
  ASSERT_EQ(sirStr1, sirStr2);
}

} // anonymous namespace
