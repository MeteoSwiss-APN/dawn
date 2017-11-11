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

// RUN: %gtclang% %file% -fno-codegen -freport-pass-preprocessor

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil Test01 {
  storage foo;

  void Do() {
    vertical_region(k_start, k_start) // EXPECTED: %line%: for(auto __k_loopvar__ : {k_start, k_start})
        foo = 1;

    vertical_region(k_start, k_end - 1) // EXPECTED: %line%: for(auto __k_loopvar__ : {k_start, k_end-1})
        foo = 1;

    vertical_region(k_start + 1, k_end - 1) // EXPECTED: %line%: for(auto __k_loopvar__ : {k_start+1, k_end-1})
        foo = 1;

    vertical_region(k_end, k_end) { // EXPECTED: %line%: for(auto __k_loopvar__ : {k_end, k_end}) {
      foo = 1;
    }
  }
};
