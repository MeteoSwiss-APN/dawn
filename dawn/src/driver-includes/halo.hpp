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
namespace dawn {

/**
 * @brief Halo extend (passed to gtclang by "-max-halo")
 * @ingroup gridtools_dawn
 */
struct halo {
#ifdef GRIDTOOLS_DAWN_HALO_EXTENT
  static constexpr int value = GRIDTOOLS_DAWN_HALO_EXTENT;
#else
  static constexpr int value = 3;
#endif
};
} // namespace dawn
} // namespace gridtools
