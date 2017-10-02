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

#ifndef DAWN_OPTIMIZER_LOOPORDER_H
#define DAWN_OPTIMIZER_LOOPORDER_H

#include <iosfwd>

namespace dawn {

/// @enum LoopOrderKind
/// @brief Loop order of a multi-stage and stages
/// @ingroup optimizer
enum class LoopOrderKind { LK_Forward = 0, LK_Backward, LK_Parallel };

/// @fn loopOrdersAreCompatible
/// @brief Check if loop order `l1` is compatible with loop order `l2` (meaning they are the same or
/// one of them is parallel)
/// @ingroup optimizer
extern bool loopOrdersAreCompatible(LoopOrderKind l1, LoopOrderKind l2);

/// @fn loopOrderToString
/// @brief Convert loop order to string
/// @ingroup optimizer
extern const char* loopOrderToString(LoopOrderKind loopOrder);

/// @brief Stream loop order
/// @ingroup optimizer
extern std::ostream& operator<<(std::ostream& os, LoopOrderKind loopOrder);

} // namespace dawn

#endif
