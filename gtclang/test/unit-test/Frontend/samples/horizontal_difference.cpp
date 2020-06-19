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

stencil horizontal_difference
{
  storage input, coeff, output;
  var lap, res, flx, fly;

  Do
  {
    vertical_region(k_start, k_end)
    {
      lap = 4.0 * input - (input[i+1] + input[i-1] + input[j+1] + input[j-1]);
      res = lap[i+1] - lap;
      flx = ((res * (input[i+1] - input)) > 0) ? 0 : res;
      res = lap[j+1] - lap;
      fly = ((res * (input[j+1] - input)) > 0) ? 0 : res;
      output = input - coeff * (flx - flx[i-1] + fly - fly[j-1]);
    }
  }
};
