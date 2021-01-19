
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

#include "dawn/IIR/Cache.h"
#include "dawn/IIR/Interval.h"
#include "dawn/IIR/MultiInterval.h"

namespace dawn {
namespace iir {

/// @brief substracts (relative complement) two intervals, generating as a result a multiinterval
MultiInterval substract(const iir::Interval& int1, const iir::Interval& int2);
/// @brief substract (relative complement) one multi-interval from an interval
MultiInterval substract(const iir::Interval& int1, const MultiInterval& int2);

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
Cache::window computeWindowOffset(LoopOrderKind loopOrder, iir::Interval const& accessInterval,
                                  iir::Interval const& computeInterval);

} // namespace iir
} // namespace dawn
