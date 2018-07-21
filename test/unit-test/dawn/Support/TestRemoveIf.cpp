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

#include "dawn/Support/RemoveIf.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

namespace dawn {

TEST(RemoveIf, Vector) {
  std::vector<int> v{1, 3, 5, 6, 7, 9};

  auto it = RemoveIf(v.begin(), v.end(), [](int a) { return a == 6; });

  // found and remove
  EXPECT_NE(it, v.end());

  EXPECT_TRUE(std::equal(v.begin(), v.end(), std::vector<int>({1, 3, 5, 7, 9}).begin()));

  it = RemoveIf(v.begin(), v.end(), [](int a) { return a == 6; });

  // not found
  EXPECT_EQ(it, v.end());

  // removing the last element
  it = RemoveIf(v.begin(), v.end(), [](int a) { return a == 9; });

  // found and remove
  EXPECT_NE(it, v.end());
}

TEST(RemoveIf, Map) {
  std::unordered_map<int, int> m{{1, 3}, {2, 4}, {3, 5}};

  bool r = RemoveIf(m.begin(), m.end(), m, [](std::pair<int, int> p) { return p.second == 4; });

  // found and remove
  EXPECT_TRUE(r);

  EXPECT_TRUE(std::equal(m.begin(), m.end(), std::unordered_map<int, int>{{1, 3}, {3, 5}}.begin()));

  r = RemoveIf(m.begin(), m.end(), m, [](std::pair<int, int> p) { return p.second == 4; });

  // not found
  EXPECT_TRUE(!r);
}

} // anonymous namespace
