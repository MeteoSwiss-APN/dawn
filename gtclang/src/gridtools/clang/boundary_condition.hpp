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

#include "gridtools/clang/stencil_function.hpp"

namespace gridtools {

namespace clang {

/*
 * @brief Boundary condition specification
 * @ingroup gridtools_clang
 */
class boundary_condition {
public:
  template <typename... T>
  boundary_condition(const stencil_function&, T&&...);
};
} // namespace clang
} // namespace gridtools
