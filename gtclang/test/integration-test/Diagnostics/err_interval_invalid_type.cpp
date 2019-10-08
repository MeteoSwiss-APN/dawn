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

stencil Test {
  storage in, out;

  Do {
    vertical_region(i + 1, i + 10) // EXPECTED_ERROR: invalid type 'struct gridtools::clang::dimension' for vertical region declaration
        in = out;
  }
};

int main() {}
