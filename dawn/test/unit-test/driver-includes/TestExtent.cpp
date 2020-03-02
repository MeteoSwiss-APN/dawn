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

#include "driver-includes/extent.hpp"

#include <gtest/gtest.h>

namespace {

TEST(driver_includes_extent, Cartesian) {
  dawn::driver::cartesian_extent e = {0, 1, 2, 3, 4, 5};

  using std::get;
  ASSERT_EQ(get<1>(e), (std::array<int, 2>{2, 3}));
  ASSERT_EQ(get<1>(get<2>(e)), 5);
}

TEST(driver_includes_extent, Unstructured) {
  dawn::driver::unstructured_extent e = {true, 1, 2};

  // a declaration of get() needs to be available via ordinary lookup until including c++17
  using std::get;
  ASSERT_TRUE(get<0>(e));

  ASSERT_EQ(get<0>(get<1>(e)), 1);
  ASSERT_EQ(get<1>(get<1>(e)), 2);
}

} // namespace
