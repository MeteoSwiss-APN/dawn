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

static const char* cacheTypeToString(Cache::CacheTypeKind cacheType) {
  switch(cacheType) {
  case Cache::CacheTypeKind::K:
    return "K";
  case Cache::CacheTypeKind::IJ:
    return "IJ";
  case Cache::CacheTypeKind::IJK:
    return "IJK";
  case Cache::CacheTypeKind::bypass:
    return "bypass";
  }
  dawn_unreachable(
      std::string("invalid cache type" + std::to_string((unsigned int)cacheType)).c_str());
}
static const char* cachePolicyToString(Cache::CacheIOPolicy cachePolicy) {
  switch(cachePolicy) {
  case Cache::CacheIOPolicy::fill:
    return "fill";
  case Cache::CacheIOPolicy::flush:
    return "flush";
  case Cache::CacheIOPolicy::local:
    return "local";
  case Cache::CacheIOPolicy::bpfill:
    return "bpfill";
  case Cache::CacheIOPolicy::epflush:
    return "epflush";
  case Cache::CacheIOPolicy::fill_and_flush:
    return "fill_and_flush";
  case Cache::CacheIOPolicy::unknown:
    return "unknown";
  }
  dawn_unreachable(
      std::string("invalid cache io policy" + std::to_string((unsigned int)cachePolicy)).c_str());
}

Cache::Cache(CacheTypeKind type, CacheIOPolicy policy, int fieldAccessID,
             const std::optional<Interval>& interval,
             const std::optional<Interval>& enclosingAccessedInterval,
             const std::optional<window>& w)
    : type_(type), policy_(policy), AccessID_(fieldAccessID), interval_(interval),
      enclosingAccessedInterval_(enclosingAccessedInterval), window_(w) {}

int Cache::getCachedFieldAccessID() const { return AccessID_; }

Interval Cache::getWindowInterval(Interval::Bound bound) const {
  DAWN_ASSERT(interval_ && window_);
  return interval_->crop(bound, {window_->m_m, window_->m_p});
}

bool Cache::requiresMemMemoryAccess() const {
  return (policy_ != CacheIOPolicy::local) || (type_ == CacheTypeKind::bypass);
}

json::json Cache::jsonDump() const {
  json::json node;
  node["accessid"] = AccessID_;
  node["type"] = cacheTypeToString(type_);
  node["policy"] = cachePolicyToString(policy_);
  std::stringstream ss;
  if(interval_) {
    ss << *interval_;
  } else {
    ss << "null";
  }
  node["interval"] = ss.str();
  ss.str("");

  if(enclosingAccessedInterval_) {
    ss << *enclosingAccessedInterval_;
  } else {
    ss << "null";
  }
  node["enclosing_accessed_interval"] = ss.str();
  ss.str("");
  if(window_) {
    ss << *window_;
  } else {
    ss << "null";
  }
  node["window"] = ss.str();
  return node;
}

std::optional<Interval> Cache::getInterval() const { return interval_; }

std::optional<Interval> Cache::getEnclosingAccessedInterval() const {
  return enclosingAccessedInterval_;
}

Cache::CacheTypeKind Cache::getCacheType() const { return type_; }

std::string Cache::getCacheTypeAsString() const {
  switch(type_) {
  case IJ:
    return "cache_type::ij";
  case K:
    return "cache_type::k";
  case IJK:
    return "cache_type::ijk";
  case bypass:
    return "cache_type::bypass";
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
