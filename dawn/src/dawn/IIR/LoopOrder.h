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

#include <iosfwd>
#include <ostream>

namespace dawn {
namespace iir {

/// @enum LoopOrderKind
/// @brief Loop order of a multi-stage and stages
/// @ingroup optimizer
enum class LoopOrderKind { Forward = 0, Backward, Parallel };

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

/// @brief increments a level according to a loop order
void increment(int& level, LoopOrderKind order);

/// @brief increments a level with certain step according to a loop order
void increment(int& lev, LoopOrderKind order, int step);

/// @brief determines whether a level < limit, where the less than comparison is performed according
/// to the loop order
bool isLevelExecBeforeEqThan(int level, int limit, LoopOrderKind order);
} // namespace iir
} // namespace dawn
