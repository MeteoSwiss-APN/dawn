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

#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil tridiagonal_solve
{
  storage inf, diag, sup, rhs, out;

  Do
  {
    vertical_region(k_start, k_start + 1)
    {
      sup = sup / diag;
      rhs = rhs / diag;
    }
    vertical_region(k_start + 1, k_end)
    {
      sup = sup / (diag - sup[k - 1] * inf);
      rhs = (rhs - inf * rhs[k - 1]) / (diag - sup[k - 1] * inf);
    }
    vertical_region(k_end - 1, k_start)
    {
      out = rhs - (sup * out[k + 1]);
    }
    vertical_region(k_end, k_end - 1)
    {
      out = rhs;
    }
  }
};
