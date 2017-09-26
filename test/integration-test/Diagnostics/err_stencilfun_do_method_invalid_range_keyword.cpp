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
  storage in, out;

  void Do(interval k_fromX = k_start, k_to = k_end) {} // EXPECTED_ERROR: invalid parameter 'k_fromX' in Do-Method specialization only 'k_from' and 'k_to' are allowed
};

stencil Test {
  storage in, out;

  Do {
    vertical_region(k_start, k_end)
      TestFun(in, out);
  }
};

int main() {}
