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
#include "gtclang/Unittest/IRSplitter.h"
#include "gtclang/Unittest/UnittestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace gtclang;

namespace {

TEST(IRSplittingTest, Interval) {
  std::string filename = "input/test_compute_read_access_interval_02.cpp";
  gtclang::IRSplitter().split(filename);
}

TEST(IRSplittingTest, RaceCondition) {
  gtclang::IRSplitter().split(
      "/home/eddied/Work/dawn/gtclang/test/integration-test/PassFieldVersioning/RaceCondition03.cpp",
      {"-freport-pass-field-versioning", "-inline=none"});
}

} // anonymous namespace
