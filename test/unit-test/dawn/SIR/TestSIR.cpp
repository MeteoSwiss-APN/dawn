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

#include "dawn/SIR/SIR.h"
#include <gtest/gtest.h>
#include <string>

using namespace dawn;

namespace {

TEST(SIRTest, Value) {
  sir::Value empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.getType(), sir::Value::None);

  // Test Boolean
  sir::Value valueBoolean;
  valueBoolean.setValue(bool(true));
  EXPECT_EQ(valueBoolean.getType(), sir::Value::Boolean);
  EXPECT_EQ(valueBoolean.getValue<bool>(), true);

  // Test Integer
  sir::Value valueInteger;
  valueInteger.setValue(int(5));
  EXPECT_EQ(valueInteger.getType(), sir::Value::Integer);
  EXPECT_EQ(valueInteger.getValue<int>(), 5);

  // Test Double
  sir::Value valueDouble;
  valueDouble.setValue(double(5.0));
  EXPECT_EQ(valueDouble.getType(), sir::Value::Double);
  EXPECT_EQ(valueDouble.getValue<double>(), 5.0);

  // Test String
  sir::Value valueString;
  valueString.setValue(std::string("string"));
  EXPECT_EQ(valueString.getType(), sir::Value::String);
  EXPECT_EQ(valueString.getValue<std::string>(), "string");
}

} // anonymous namespace
