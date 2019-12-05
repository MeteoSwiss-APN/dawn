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

namespace gtclang {
namespace dsl {

/**
 * @brief Defintion of an offset which can be as argument in stencil functions
 * @ingroup gtclang_dsl
 */
struct offset {
  offset operator+(int);
  offset operator-(int);

  offset operator+();
  offset operator-();
};
} // namespace dsl
} // namespace gtclang
