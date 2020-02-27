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

TEST(IRSplittingTest, FieldVersioning) {
  gtclang::IRSplitter("../../../../dawn/test/unit-test/dawn/Optimizer/Passes/samples", 1)
      .split("../../../../dawn/test/unit-test/dawn/Optimizer/Passes/samples/RaceCondition01.cpp",
             {"-freport-pass-field-versioning"});
}

TEST(IRSplittingTest, StageReordering) {
  gtclang::IRSplitter("../../../../dawn/test/unit-test/dawn/Optimizer/Passes", 1)
      .split("../../../../dawn/test/unit-test/dawn/Optimizer/Passes/samples/ReorderTest07.cpp",
             {"-freport-pass-stage-reordering"});
}

TEST(IRSplittingTest, CacheTest) {
  gtclang::IRSplitter("../../../dawn/test/unit-test/dawn/Optimizer/Passes", 100).split(
      "PassSetCaches/IJCacheTest02.cpp", {"-freport-pass-set-caches"});
}

TEST(IRSplittingTest, TemporaryMerger) {
  gtclang::IRSplitter("dawn/test/unit-test/dawn/Optimizer/Passes", 3)
      .split("dawn/test/unit-test/dawn/Optimizer/Passes/samples/MergeTest05.cpp",
             {"-fmerge-temporaries"});
}

} // anonymous namespace
