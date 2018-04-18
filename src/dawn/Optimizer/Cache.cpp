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
             const boost::optional<Interval>& interval)
    : type_(type), policy_(policy), AccessID_(fieldAccessID), interval_(interval) {}

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

} // namespace dawn
