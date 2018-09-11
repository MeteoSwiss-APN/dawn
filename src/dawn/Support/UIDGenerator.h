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

#ifndef DAWN_SUPPORT_UIDGENERATOR
#define DAWN_SUPPORT_UIDGENERATOR

#include "dawn/Support/NonCopyable.h"

namespace dawn {

/// @brief Unique identifier generator (starting from @b 1)
/// @ingroup support
class UIDGenerator : NonCopyable {
  int counter_;
  static UIDGenerator* instance_;

  UIDGenerator() : counter_(1) {}

public:
  static UIDGenerator* getInstance();

  /// @brief Get a unique *strictly* positive identifer
  int get() { return (counter_++); }
};

} // namespace dawn

#endif
