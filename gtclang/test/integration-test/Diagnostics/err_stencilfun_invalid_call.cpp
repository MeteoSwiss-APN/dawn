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

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gridtools::clang;

stencil_function TestFun {
  storage in;
  storage out;

  Do() {}
};

stencil Test {
  storage in;
  storage out;

  Do {
    vertical_region(k_start, k_end)
        TestFun(in, out, out); // EXPECTED_ERROR: no matching stencil function for call to 'TestFun'
  }
};

int main() {}
