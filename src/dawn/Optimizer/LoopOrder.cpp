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

#include "dawn/Optimizer/LoopOrder.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>

namespace dawn {

bool loopOrdersAreCompatible(LoopOrderKind l1, LoopOrderKind l2) {
  return (l1 == l2 || l1 == LoopOrderKind::LK_Parallel || l2 == LoopOrderKind::LK_Parallel);
}

std::ostream& operator<<(std::ostream& os, LoopOrderKind loopOrder) {
  return (os << loopOrderToString(loopOrder));
}

const char* loopOrderToString(LoopOrderKind loopOrder) {
  switch(loopOrder) {
  case LoopOrderKind::LK_Forward:
    return "forward";
  case LoopOrderKind::LK_Backward:
    return "backward";
  case LoopOrderKind::LK_Parallel:
    return "parallel";
  }
  dawn_unreachable("invalid loop order");
}

} // namespace dawn
