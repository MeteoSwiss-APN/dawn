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

#include "dawn-c/Options.h"
#include <cstring>
#include <gtest/gtest.h>

namespace {

TEST(OptionsTest, OptionsEntryInteger) {
  dawnOptionsEntry_t* entry = dawnOptionsEntryCreateInteger(5);
  EXPECT_EQ(entry->Type, DT_Integer);
  EXPECT_EQ(entry->SizeInBytes, sizeof(int));

  int* value = dawnOptionsEntryGetInteger(entry);
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 5);

  double* invalid = dawnOptionsEntryGetDouble(entry);
  ASSERT_EQ(invalid, nullptr);

  std::free(value);
  dawnOptionsEntryDestroy(entry);
}

TEST(OptionsTest, OptionsEntryDouble) {
  dawnOptionsEntry_t* entry = dawnOptionsEntryCreateDouble(5.5);
  EXPECT_EQ(entry->Type, DT_Double);
  EXPECT_EQ(entry->SizeInBytes, sizeof(double));

  double* value = dawnOptionsEntryGetDouble(entry);
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 5.5);

  std::free(value);
  dawnOptionsEntryDestroy(entry);
}

TEST(OptionsTest, OptionsEntryString) {
  dawnOptionsEntry_t* entry = dawnOptionsEntryCreateString("hello");
  EXPECT_EQ(entry->Type, DT_Char);
  EXPECT_EQ(entry->SizeInBytes, std::strlen("hello") + 1);

  char* value = dawnOptionsEntryGetString(entry);
  ASSERT_NE(value, nullptr);
  EXPECT_STREQ(value, "hello");

  std::free(value);
  dawnOptionsEntryDestroy(entry);
}

TEST(OptionsTest, OptionsCreateAndDestroy) {
  dawnOptions_t* options = dawnOptionsCreate();
  dawnOptionsDestroy(options);
}

TEST(OptionsTest, OptionsSetAndGetInteger) {
  dawnOptions_t* options = dawnOptionsCreate();

  dawnOptionsEntry_t* entry = dawnOptionsEntryCreateInteger(5);
  dawnOptionsSet(options, "int", entry);
  dawnOptionsEntryDestroy(entry);

  entry = dawnOptionsGet(options, "int");
  int* value = dawnOptionsEntryGetInteger(entry);
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 5);
  std::free(value);
  dawnOptionsEntryDestroy(entry);

  dawnOptionsDestroy(options);
}

TEST(OptionsTest, OptionsSetAndGetDouble) {
  dawnOptions_t* options = dawnOptionsCreate();

  dawnOptionsEntry_t* entry = dawnOptionsEntryCreateDouble(5.5);
  dawnOptionsSet(options, "double", entry);
  dawnOptionsEntryDestroy(entry);

  entry = dawnOptionsGet(options, "double");
  double* value = dawnOptionsEntryGetDouble(entry);
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 5.5);
  std::free(value);
  dawnOptionsEntryDestroy(entry);

  dawnOptionsDestroy(options);
}

TEST(OptionsTest, OptionsSetAndGetString) {
  dawnOptions_t* options = dawnOptionsCreate();

  dawnOptionsEntry_t* entry = dawnOptionsEntryCreateString("hello");
  dawnOptionsSet(options, "string", entry);
  dawnOptionsEntryDestroy(entry);

  entry = dawnOptionsGet(options, "string");
  char* value = dawnOptionsEntryGetString(entry);
  ASSERT_NE(value, nullptr);
  EXPECT_STREQ(value, "hello");
  std::free(value);
  dawnOptionsEntryDestroy(entry);

  dawnOptionsDestroy(options);
}

TEST(OptionsTest, OptionsToString) {
  dawnOptions_t* options = dawnOptionsCreate();

  char* str = dawnOptionsToString(options);
  EXPECT_NE(str, nullptr);
  std::free(str);

  dawnOptionsDestroy(options);
}

} // anonymous namespace
