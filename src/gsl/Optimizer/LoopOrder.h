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

#ifndef GSL_OPTIMIZER_LOOPORDER_H
#define GSL_OPTIMIZER_LOOPORDER_H

#include <iosfwd>

namespace gsl {

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

} // namespace gsl

#endif
