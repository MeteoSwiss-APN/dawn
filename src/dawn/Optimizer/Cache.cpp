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

#include "dawn/Optimizer/Cache.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {

Cache::Cache(CacheTypeKind type, CacheIOPolicy policy, int fieldAccessID,
             const boost::optional<Interval>& interval, const boost::optional<window>& w)
    : type_(type), policy_(policy), AccessID_(fieldAccessID), interval_(interval), window_(w) {}

int Cache::getCachedFieldAccessID() const { return AccessID_; }

boost::optional<Interval> Cache::getInterval() const { return interval_; }

Cache::CacheTypeKind Cache::getCacheType() const { return type_; }

std::string Cache::getCacheTypeAsString() const {
  switch(type_) {
  case IJ:
    return "IJ";
  case K:
    return "K";
  case IJK:
    return "IJK";
  case bypass:
    return "bypass";
  }
  dawn_unreachable("invalid cache type");
}

Cache::CacheIOPolicy Cache::getCacheIOPolicy() const { return policy_; }

std::string Cache::getCacheIOPolicyAsString() const {
  switch(policy_) {
  case fill_and_flush:
    return "fill_and_flush";
  case fill:
    return "fill";
  case flush:
    return "flush";
  case epflush:
    return "epflush";
  case bpfill:
    return "bpfill";
  case local:
    return "local";
  default:
    dawn_unreachable("invalid cache type");
  }
}

std::ostream& operator<<(std::ostream& os, Cache::window const& w) {
  return os << "window[" << std::to_string(w.m_m) << "," << std::to_string(w.m_p) << "]";
}

bool operator==(const Cache::window& first, const Cache::window& second) {
  return ((first.m_m == second.m_m) && (first.m_p == second.m_p));
}

} // namespace dawn
