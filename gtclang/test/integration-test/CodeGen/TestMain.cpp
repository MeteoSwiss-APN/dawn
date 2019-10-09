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
#include <gtest/gtest.h>
#include "test/integration-test/CodeGen/Options.hpp"

using namespace dawn;

int main(int argc, char** argv) {
  // Pass command line arguments to googltest
  ::testing::InitGoogleTest(&argc, argv);

  if(argc < 4) {
    printf("Usage: <pack>_stencil_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields\n");
    return 1;
  }

  for(int i = 0; i != 3; ++i) {
    Options::getInstance().m_size[i] = atoi(argv[i + 1]);
  }
  return RUN_ALL_TESTS();
}
