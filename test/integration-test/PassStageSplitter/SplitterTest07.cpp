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

// RUN: %gtclang% %file% -fno-codegen -freport-pass-stage-split

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil Test {
  storage field_a, field_b, field_c, field_d, field_e;

  Do {
    vertical_region(k_start, k_end) {
      field_d = field_e(i + 1); // EXPECTED: PASS: PassStageSplitter: Test: split:%line%
      field_c = field_d(i + 1); // EXPECTED: PASS: PassStageSplitter: Test: split:%line%
      field_b = field_c(i + 1); // EXPECTED: PASS: PassStageSplitter: Test: split:%line%
      field_a = field_b(i + 1);
    }
  }
};

int main() {}
