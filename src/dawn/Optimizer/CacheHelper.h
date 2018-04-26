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

#ifndef DAWN_OPTIMIZER_CACHE_HELPER_H
#define DAWN_OPTIMIZER_CACHE_HELPER_H

#include "dawn/Optimizer/Cache.h"
#include <boost/optional.hpp>

namespace dawn {

namespace cache {
struct window {
  int m_m, m_p;
};

struct CacheHelper {
  static boost::optional<window> ComputeBPFillCacheWindow(MultiStage const& ms,
                                                          const int accessID) {

    return boost::optional<window>();
  }
};

} // namespace cache
} // namespace dawn

#endif
