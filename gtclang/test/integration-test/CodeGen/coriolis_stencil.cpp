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

using namespace gridtools::clang;

stencil coriolis_stencil {
  storage u_tens, u_nnow, v_tens, v_nnow, fc;

  Do {
    vertical_region(k_start, k_end) {
      double z_fv_north = fc * (v_nnow + v_nnow(i + 1));
      double z_fv_south = fc(j - 1) * (v_nnow(j - 1) + v_nnow(i + 1, j - 1));
      u_tens += 0.25 * (z_fv_north + z_fv_south);

      double z_fu_east = fc * (u_nnow + u_nnow(j + 1));
      double z_fu_west = fc(i - 1) * (u_nnow(i - 1) + u_nnow(i - 1, j + 1));
      v_tens -= 0.25 * (z_fu_east + z_fu_west);
    }
  }
};
