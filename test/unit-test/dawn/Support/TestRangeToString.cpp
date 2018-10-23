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

#include "dawn/Support/StringUtil.h"
#include <gtest/gtest.h>

namespace dawn {

TEST(RangeToString, default) {
  RangeToString rangeToString;
  EXPECT_EQ((rangeToString(std::vector<int>{2, 3, 4})), "[2, 3, 4]");
}

TEST(RangeToString, limits) {
  RangeToString rangeToString("+", "l", "r");
  EXPECT_EQ((rangeToString(std::vector<int>{2, 3, 4})), "l2+3+4r");
}

TEST(RangeToString, space) {
  RangeToString rangeToString("+", "{", "}");
  EXPECT_EQ((rangeToString(std::vector<std::string>{"2", " ", "4"})), "{2+ +4}");
}
TEST(RangeToString, empty) {
  RangeToString rangeToString("+", "{", "}");
  EXPECT_EQ((rangeToString(std::vector<std::string>{"2", "", "4"})), "{2++4}");
}

TEST(RangeToString, ignoreIfEmpty) {
  RangeToString rangeToString("+", "{", "}", true);
  EXPECT_EQ((rangeToString(std::vector<std::string>{"2", "", "4"})), "{2+4}");
}
TEST(RangeToString, ignoreIfEmptyF) {
  RangeToString rangeToString("+", "{", "}", true);
  EXPECT_EQ((rangeToString(std::vector<std::string>{"", "3", "4"})), "{3+4}");
}
TEST(RangeToString, ignoreIfEmptyL) {
  RangeToString rangeToString("+", "{", "}", true);
  EXPECT_EQ((rangeToString(std::vector<std::string>{"2", "3", ""})), "{2+3}");
}

} // namespace dawn
