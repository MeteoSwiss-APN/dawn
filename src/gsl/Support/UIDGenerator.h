//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_SUPPORT_UIDGENERATOR
#define GSL_SUPPORT_UIDGENERATOR

#include "gsl/Support/NonCopyable.h"

namespace gsl {

/// @brief Unique identifier generator (starting from @b 1)
/// @ingroup support
class UIDGenerator : NonCopyable {
  int counter_;

public:
  UIDGenerator() : counter_(1) {}

  /// @brief Get a unique *strictly* positive identifer
  int get() { return (counter_++); }
};

} // namespace gsl

#endif
