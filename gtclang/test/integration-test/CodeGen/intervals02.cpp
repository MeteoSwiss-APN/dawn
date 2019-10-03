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

#ifndef GRIDTOOLS_CLANG_GENERATED
interval k_flat = k_start + 4;
#endif

stencil intervals02 {
  storage in, out;

  Do {
      vertical_region(k_start+1, k_start+1)
          out = 0;

      vertical_region(k_start+2, k_flat)
          out = out[k-1] + in + 1;

      //TODO add protection that k_end covers at least all the k_flat and vertical regions used
      vertical_region(k_flat + 2, k_flat+2)
          out = 0;
      vertical_region(k_flat + 3, k_end-1)
          out = out[k-1] + in + 3;
  }
};
