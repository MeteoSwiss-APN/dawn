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

#include "dawn/Support/IndexRange.h"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace dawn {

TEST(IndexRange, VectorInt) {
  std::vector<int> v{1, 3, 5, 6, 7, 9};
  auto loopRange = makeRange(v, [](int const& val) { return val < 6; });

  ASSERT_TRUE((loopRange.size() == 3));
  for(auto vIt : loopRange) {
    ASSERT_TRUE((*vIt == v[vIt.idx()]));
  }
}

TEST(IndexRange, VectorIntModify) {
  std::vector<int> v{1, 3, 5, 6, 7, 9};
  auto loopRange = makeRange(v, [](int const& val) { return val < 6; });

  ASSERT_TRUE((loopRange.size() == 3));
  for(auto vIt : loopRange) {
    *vIt += 3;
  }

  std::vector<int> vref{4, 6, 8, 6, 7, 9};
  for(auto vIt : loopRange) {
    ASSERT_TRUE((*vIt == vref[vIt.idx()]));
  }
}

TEST(IndexRange, ConstVectorInt) {
  const std::vector<int> v{1, 3, 5, 6, 7, 9};
  auto loopRange = makeRange(v, [](int const& val) { return val < 6; });

  ASSERT_TRUE((loopRange.size() == 3));
  for(auto vIt : loopRange) {
    ASSERT_TRUE((*vIt == v[vIt.idx()]));
  }
}

} // anonymous namespace
