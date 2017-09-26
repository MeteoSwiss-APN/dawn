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

#include "gsl/Optimizer/LoopOrder.h"
#include "gsl/Support/Unreachable.h"
#include <iostream>

namespace gsl {

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
  gsl_unreachable("invalid loop order");
}

} // namespace gsl
