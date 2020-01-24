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

#include "dawn/Support/IteratorAdapters.h"
#include <gtest/gtest.h>
#include <vector>

using namespace dawn;

namespace {

TEST(TestIteratorAdapters, reverse) {
  const std::vector<int> vec{0, 1, 2};

  int i = 0;
  for(auto e : reverse(vec)) {
    switch(i) {
    case 0:
      ASSERT_EQ(e, 2);
      break;
    case 1:
      ASSERT_EQ(e, 1);
      break;
    case 2:
      ASSERT_EQ(e, 0);
      break;
    }
    ++i;
  }
}

TEST(TestIteratorAdapters, enumerate) {
  const std::vector<int> vec{0, 1, 2};

  for(auto [i, e] : enumerate(vec)) {
    switch(i) {
    case 0:
      ASSERT_EQ(e, 0);
      ASSERT_EQ(i, 0);
      break;
    case 1:
      ASSERT_EQ(e, 1);
      ASSERT_EQ(i, 1);
      break;
    case 2:
      ASSERT_EQ(e, 2);
      ASSERT_EQ(i, 2);
      break;
    }
  }
}

} // anonymous namespace
