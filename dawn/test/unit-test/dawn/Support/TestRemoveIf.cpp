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
#include <unordered_map>
#include <vector>

namespace dawn {

TEST(RemoveIf, Vector) {
  std::vector<int> v{1, 3, 5, 6, 7, 9};

  bool found = RemoveIf(v, [](int a) { return a == 6; });

  // found and remove
  EXPECT_TRUE(found);
  EXPECT_EQ(v.size(), 5);

  EXPECT_TRUE(std::equal(v.begin(), v.end(), std::vector<int>({1, 3, 5, 7, 9}).begin()));

  found = RemoveIf(v, [](int a) { return a == 6; });

  // not found
  EXPECT_TRUE(!found);
  EXPECT_EQ(v.size(), 5);

  // removing the last element
  found = RemoveIf(v, [](int a) { return a == 9; });

  // found and remove
  EXPECT_TRUE(found);
  EXPECT_EQ(v.size(), 4);
}

TEST(RemoveIf, Map) {
  std::unordered_map<int, int> m{{1, 3}, {2, 4}, {3, 5}};

  bool r = RemoveIf(m, [](std::pair<int, int> p) { return p.second == 4; });

  // found and remove
  EXPECT_TRUE(r);
  EXPECT_EQ(m.size(), 2);

  EXPECT_TRUE(std::equal(m.begin(), m.end(), std::unordered_map<int, int>{{1, 3}, {3, 5}}.begin()));

  r = RemoveIf(m, [](std::pair<int, int> p) { return p.second == 4; });

  // not found
  EXPECT_TRUE(!r);
  EXPECT_EQ(m.size(), 2);
}

} // namespace dawn
