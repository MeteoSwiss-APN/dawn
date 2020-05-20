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

#include "dawn/IIR/LoopOrder.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace iir {

bool loopOrdersAreCompatible(LoopOrderKind l1, LoopOrderKind l2) {
  return (l1 == l2 || l1 == LoopOrderKind::Parallel || l2 == LoopOrderKind::Parallel);
}

std::ostream& operator<<(std::ostream& os, LoopOrderKind loopOrder) {
  return (os << loopOrderToString(loopOrder));
}

const char* loopOrderToString(LoopOrderKind loopOrder) {
  switch(loopOrder) {
  case LoopOrderKind::Forward:
    return "forward";
  case LoopOrderKind::Backward:
    return "backward";
  case LoopOrderKind::Parallel:
    return "parallel";
  }
  dawn_unreachable(
      std::string("invalid loop order" + std::to_string((unsigned int)loopOrder)).c_str());
}

void increment(int& lev, LoopOrderKind order) {
  if(order == LoopOrderKind::Backward) {
    lev--;
  } else {
    lev++;
  }
}

void increment(int& lev, LoopOrderKind order, int step) {
  if(order == LoopOrderKind::Backward) {
    lev -= step;
  } else {
    lev += step;
  }
}

bool isLevelExecBeforeEqThan(int level, int limit, LoopOrderKind order) {
  if(order == LoopOrderKind::Backward) {
    if(level >= limit) {
      return true;
    }
    return false;
  } else {
    if(level <= limit) {
      return true;
    }
    return false;
  }
}

} // namespace iir
} // namespace dawn
