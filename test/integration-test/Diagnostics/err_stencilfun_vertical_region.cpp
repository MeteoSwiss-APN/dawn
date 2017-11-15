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

// RUN: %gtclang% %file% -fno-codegen

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil_function TestFun {
  storage in;
  storage out;

  Do {
    vertical_region(k_start, k_end) // EXPECTED_ERROR: invalid vertical region in stencil function
        in = out;
  }
};

stencil Test {
  storage in;
  storage out;

  void Do() {
    vertical_region(k_start, k_end)
        in = out;
  }
};

int main() {}
