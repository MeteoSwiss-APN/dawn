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
#include "dawn/SIR/SIRSerializer.h"
#include "dawn/Unittest/PrintAllExpressionTypes.h"
#include <gtest/gtest.h>

using namespace dawn;

namespace {

TEST(SIRSerializer, Serialize) {
  SIR sir;
  sir.Filename = "test";
  
  std::cout << SIRSerializer::serializeToString(&sir) << std::endl;
}

TEST(SIRSerializer, Deserialize) {

}

} // anonymous namespace
