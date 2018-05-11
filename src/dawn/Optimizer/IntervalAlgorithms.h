
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

MultiInterval substract(const Interval& int1, const Interval& int2);

Cache::window computeWindowOffset(LoopOrderKind loopOrder, Interval const& accessInterval,
                                  Interval const& computeInterval);

} // namespace dawn

#endif
