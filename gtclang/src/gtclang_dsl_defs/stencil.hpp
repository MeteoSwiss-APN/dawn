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

#pragma once

#include "gtclang_dsl_defs/dimension.hpp"

namespace gtclang {
namespace dsl {

/**
 * @brief A runnable stencil
 * @ingroup gridtools_clang
 */
class stencil {
protected:
  dimension i;
  dimension j;
  dimension k;

public:
  template <typename... T>
  stencil(T&&...);

  /**
   * @brief Invoke the stencil program by calling the individual stencils
   *
   * @param make_steady   Prepare the stencil for execuation first (calls `make_steady`)
   */
  void run(bool make_steady = true);

  /**
   * @brief Prepare the stencil for execuation by copying all fields to the device
   */
  void make_steady();
};
} // namespace dsl
} // namespace gtclang
