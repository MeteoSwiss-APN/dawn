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

stencil_function delta {
  direction dir;
  storage data;
  Do { return data[dir + 1] - data; }
};

stencil compute_extent_test_stencil {
  storage u, out, coeff;

  var flx, fly, lap, lap2;
  Do {
    vertical_region(k_start, k_end) {
      lap = u[i + 1] + u[i - 1] + u[j + 1] + u[j - 1] - 4.0 * u;
      if(flx * delta(i, u) > 0)
        flx = 0.;
      else
        flx = lap[i + 1] - lap;

      if(fly * delta(j, u) > 0)
        fly = 0.;
      else
        fly = delta(j, lap);
      lap2 = u - coeff * (flx - flx[i - 1] + fly - fly[j - 1]);

      out = lap2[j + 1] - lap2;
    }
  }
};
