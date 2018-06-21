
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

#ifndef DAWN_OPTIMIZER_INTERVALALGORITHMS_H
#define DAWN_OPTIMIZER_INTERVALALGORITHMS_H

#include "dawn/Optimizer/Interval.h"
#include "dawn/Optimizer/MultiInterval.h"
#include "dawn/Optimizer/Cache.h"

namespace dawn {

/// @brief substracts (relative complement) two intervals, generating as a result a multiinterval
MultiInterval substract(const Interval& int1, const Interval& int2);
/// @brief substract (relative complement) one multi-interval from an interval
MultiInterval substract(const Interval& int1, const MultiInterval& int2);

/// @brief computes the window offset of a kcache concept
/// @param loopOrder is the loop order of the vertical execution
/// @param accessInterval interval where the data has been accessed in main memory (i.e. discarding
/// accesses to data that is computed within the stencil in iterative solvers).
///
/// Example:
/// Do(kstart,kend)
/// {return u += u[k+1];} the interval is {kend+1, kend+1}
/// @param computeInterval interval of the iteration space, i.e. for the previous example the
/// interval is {kstart, kend}
Cache::window computeWindowOffset(LoopOrderKind loopOrder, Interval const& accessInterval,
                                  Interval const& computeInterval);

} // namespace dawn

#endif
