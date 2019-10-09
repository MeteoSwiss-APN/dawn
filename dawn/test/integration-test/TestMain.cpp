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

#include "dawn/Support/STLExtras.h"
#include "dawn/Unittest/UnittestLogger.h"
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {

  // Initialize gtest
  testing::InitGoogleTest(&argc, argv);

  // Initialize Unittest-Logger
  auto logger = std::make_unique<dawn::UnittestLogger>();
  dawn::Logger::getSingleton().registerLogger(logger.get());

  return RUN_ALL_TESTS();
}
