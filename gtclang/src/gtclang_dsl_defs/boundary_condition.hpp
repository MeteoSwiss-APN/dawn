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

#include "gtclang_dsl_defs/stencil_function.hpp"

namespace gtclang {
namespace dsl {
/*
 * @brief Boundary condition specification
 * @ingroup gtclang_dsl
 */
class boundary_condition {
public:
  template <typename... T>
  boundary_condition(const stencil_function&, T&&...);
};
} // namespace dsl
} // namespace gtclang
