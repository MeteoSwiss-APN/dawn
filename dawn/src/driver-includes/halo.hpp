//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#pragma once

namespace gridtools {

namespace clang {

/**
 * @brief Halo extend (passed to gtclang by "-max-halo")
 * @ingroup gridtools_clang
 */
struct halo {
#ifdef GRIDTOOLS_CLANG_HALO_EXTEND
  static constexpr int value = GRIDTOOLS_CLANG_HALO_EXTEND;
#else
  static constexpr int value = 3;
#endif
};
} // namespace clang
} // namespace gridtools
