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

#include "gridtools/clang/dimension.hpp"

namespace gridtools {

namespace clang {

/*
 * @brief A stencil which can be called as a function from other `stencils`
 * @ingroup gridtools_clang
 */
class stencil_function {
protected:
  dimension i;
  dimension j;
  dimension k;

private:
  stencil_function& operator=(stencil_function) = delete;

public:
  template <typename... T>
  stencil_function(T&&...);

  operator double() const;
};
} // namespace clang
} // namespace gridtools
