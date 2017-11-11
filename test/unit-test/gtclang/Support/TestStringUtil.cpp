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

#include "gtclang/Support/StringUtil.h"
#include <gtest/gtest.h>

using namespace gtclang;

namespace {

TEST(StringUtilTest, SplitString) {
  std::string str("The quick brown fox jumps over the lazy dog.");

  // Split over 20 charachters
  std::string resStr1(splitString(str, 20, 0));
  std::string refStr1("The quick brown fox\njumps over the lazy\ndog.");
  EXPECT_STREQ(refStr1.c_str(), resStr1.c_str());

  // Split over 25 characters and indent *every* line by 5 characters
  std::string resStr2(splitString(str, 25, 5, true));
  std::string refStr2("     The quick brown fox\n     jumps over the lazy\n     dog.");
  EXPECT_STREQ(refStr2.c_str(), resStr2.c_str());

  // Split over 25 characters and indent *new* lines by 5 characters
  std::string resStr3(splitString(str, 25, 5, false));
  std::string refStr3("The quick brown fox\n     jumps over the lazy\n     dog.");
  EXPECT_STREQ(refStr3.c_str(), resStr3.c_str());

  // Split over 80 characters (should do nothing here)
  std::string resStr4(splitString(str, 80, 0));
  EXPECT_STREQ(str.c_str(), resStr4.c_str());

  // Split over 80 characters and indent lines by 40
  std::string resStr5(splitString(str, 80, 40));
  std::string refStr5("                                        The quick brown fox jumps over "
                      "the lazy\n                                        dog.");
  EXPECT_STREQ(refStr5.c_str(), resStr5.c_str());

  // Split long words over 10
  std::string str6("ThisIsTooLongToBeSpread ThisIsAsWell");
  std::string refStr6("ThisIsTooLongToBeSpread\nThisIsAsWell");
  std::string resStr6(splitString(str6, 10, 0));
  EXPECT_STREQ(refStr6.c_str(), resStr6.c_str());

  // Handle internal '\n'
  std::string str7("The quick\n brown fox jumps over the lazy\n dog.");
  std::string refStr7("     The quick\n     brown fox jumps\n     over the lazy\n     dog.");
  std::string resStr7(splitString(str7, 25, 5, true));
  EXPECT_STREQ(refStr7.c_str(), resStr7.c_str());
}

TEST(StringUtilTest, TokenizeString) {

  // Tokenize with ' '
  std::string str1("The quick brown fox jumps over the lazy dog.");
  EXPECT_EQ(tokenizeString(str1, " "),
            (std::vector<std::string>{"The", "quick", "brown", "fox", "jumps", "over", "the",
                                      "lazy", "dog."}));

  // Tokeninze with ','
  std::string str2("The,quick,brown,fox,   jumps, over, the,lazy , dog.");
  EXPECT_EQ(tokenizeString(str2, ","),
            (std::vector<std::string>{"The", "quick", "brown", "fox", "   jumps", " over", " the",
                                      "lazy ", " dog."}));

  // Tokeninze with ',:|'
  std::string str3("The,quick:brown|fox");
  EXPECT_EQ(tokenizeString(str3, ",:|"),
            (std::vector<std::string>{"The", "quick", "brown", "fox"}));

  // Tokeninze with '\n'
  std::string str4("\nA");
  EXPECT_EQ(tokenizeString(str4, "\n"), (std::vector<std::string>{"A"}));
}

} // anonymous namespace
