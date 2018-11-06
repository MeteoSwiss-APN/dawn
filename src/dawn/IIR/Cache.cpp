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

#include "dawn/IIR/Cache.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace iir {

Cache::Cache(CacheTypeKind type, CacheIOPolicy policy, int fieldAccessID,
             const boost::optional<Interval>& interval,
             const boost::optional<Interval>& enclosingAccessedInterval,
             const boost::optional<window>& w)
    : type_(type), policy_(policy), AccessID_(fieldAccessID), interval_(interval),
      enclosingAccessedInterval_(enclosingAccessedInterval), window_(w) {}

int Cache::getCachedFieldAccessID() const { return AccessID_; }

boost::optional<Interval> Cache::getInterval() const { return interval_; }

boost::optional<Interval> Cache::getEnclosingAccessedInterval() const {
  return enclosingAccessedInterval_;
}

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

bool Cache::requiresWindow() const {
  return getCacheIOPolicy() == Cache::bpfill || getCacheIOPolicy() == Cache::epflush;
}

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
  return os << "window" << w.toString();
}

bool operator==(const Cache::window& first, const Cache::window& second) {
  return ((first.m_m == second.m_m) && (first.m_p == second.m_p));
}

} // namespace iir
} // namespace dawn
