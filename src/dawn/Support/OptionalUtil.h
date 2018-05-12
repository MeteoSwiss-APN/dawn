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

#ifndef DAWN_SUPPORT_OPTIONALUTIL_H
#define DAWN_SUPPORT_OPTIONALUTIL_H

#include <boost/optional.hpp>

namespace dawn {

template <typename T, typename F>
inline boost::optional<T> operateOnOptionals(boost::optional<T> t1, boost::optional<T> t2,
                                             F const& op) {
  if(!t1.is_initialized())
    return t2;
  if(!t2.is_initialized())
    return t1;
  return op(t1, t2);
}

} // namespace dawn

#endif
