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

#include "dawn/Support/STLExtras.h"
#include "gtclang/Support/Logger.h"
#include "gtclang/Unittest/UnittestEnvironment.h"
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {

  // Initialize GTest
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(&gtclang::UnittestEnvironment::getSingleton());

  return RUN_ALL_TESTS();
}
