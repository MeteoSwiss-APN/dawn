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
#include <ostream>
#include <vector>

namespace dawn {

TEST(IndexRange, VectorInt) {
  std::vector<int> v{1, 3, 5, 6, 7, 9};
  auto loopRange = makeRange(v, [](int const& val) { return val < 6; });

  ASSERT_TRUE((loopRange.size() == 3));
  for(auto vIt = loopRange.begin(); vIt != loopRange.end(); ++vIt) {
    ASSERT_TRUE((*vIt == v[vIt.idx()]));
  }
}

TEST(IndexRange, ItSub) {
  std::vector<int> v{1, 3, 9, 13, 6, 7};
  auto loopRange = makeRange(v, [](int const& val) { return val < 10; });

  ASSERT_TRUE((loopRange.size() == 5));
  std::vector<int> vref{1, 3, 9, 6, 7};

  auto vIt = loopRange.end();
  for(--vIt; vIt != loopRange.begin(); --vIt) {
    ASSERT_TRUE((*vIt == vref[vIt.idx()]));
  }
  // test for begin
  ASSERT_TRUE((*vIt == vref[vIt.idx()]));
}

TEST(IndexRange, VectorIntModify) {
  std::vector<int> v{1, 3, 9, 13, 6, 7};
  auto loopRange = makeRange(v, [](int const& val) { return val < 10; });

  ASSERT_TRUE((loopRange.size() == 5));

  std::vector<int> vref{1, 3, 9, 6, 7};
  int cnt = 0;
  for(auto vIt = loopRange.begin(); vIt != loopRange.end(); ++vIt, ++cnt) {
    ASSERT_TRUE(vIt.idx() == cnt);
    ASSERT_TRUE((*vIt == vref[vIt.idx()]));
  }

  for(auto& val : loopRange) {
    val += 3;
  }

  std::vector<int> vref2{4, 6, 9};
  for(auto vIt = loopRange.begin(); vIt != loopRange.end(); ++vIt) {
    ASSERT_TRUE((*vIt == vref2[vIt.idx()]));
  }
}

TEST(IndexRange, ConstVectorInt) {
  const std::vector<int> v{1, 3, 5, 6, 7, 9};
  auto loopRange = makeRange(v, [](int const& val) { return val < 6; });

  ASSERT_TRUE((loopRange.size() == 3));
  for(auto vIt = loopRange.begin(); vIt != loopRange.end(); ++vIt) {
    ASSERT_TRUE((*vIt == v[vIt.idx()]));
  }
}

} // namespace dawn
