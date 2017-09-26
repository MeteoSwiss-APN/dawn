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

#include "dawn/Support/Type.h"
#include <gtest/gtest.h>
#include <sstream>

using namespace dawn;

namespace {

TEST(TypeTest, Construction) {
  Type t0{BuiltinTypeID::Float, CVQualifier::Const | CVQualifier::Volatile};
  EXPECT_TRUE(t0.isBuiltinType());
  EXPECT_TRUE(t0.isConst());
  EXPECT_TRUE(t0.isVolatile());

  Type t1{"foo", CVQualifier::None};
  EXPECT_FALSE(t1.isBuiltinType());
  EXPECT_FALSE(t1.isConst());
  EXPECT_FALSE(t1.isVolatile());
}

TEST(TypeTest, Stringify) {
  {
    Type t0{BuiltinTypeID::Float, CVQualifier::Const | CVQualifier::Volatile};

    std::stringstream ss;
    ss << t0;
    EXPECT_STREQ(ss.str().c_str(), "const volatile float_type");
  }

  {
    Type t1{"foo", CVQualifier::None};
    std::stringstream ss;
    ss << t1;
    EXPECT_STREQ(ss.str().c_str(), "foo");
  }
}

} // anonymous namespace
