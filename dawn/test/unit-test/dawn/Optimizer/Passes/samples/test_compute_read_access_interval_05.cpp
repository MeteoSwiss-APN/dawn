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

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

stencil stencil {
  storage dcol, ccol, t_nnow, tp_tens, datacol;
  //  var datacol;
  Do {
    //    vertical_region(k_end, k_end)
    //       datacol=0;

    vertical_region(k_end - 1, k_start) {

      double retval = dcol - ccol * datacol[k + 1];
      datacol = retval;

      tp_tens = (datacol - t_nnow) * 1.2;
    }
  }
};
