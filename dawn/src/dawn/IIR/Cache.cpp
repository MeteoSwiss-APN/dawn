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

static const char* cacheTypeToString(Cache::TypeKind cacheType) {
  switch(cacheType) {
  case Cache::TypeKind::K:
    return "K";
  case Cache::TypeKind::IJ:
    return "IJ";
  case Cache::TypeKind::IJK:
    return "IJK";
  case Cache::TypeKind::bypass:
    return "bypass";
  }
  dawn_unreachable(
      std::string("invalid cache type" + std::to_string((unsigned int)cacheType)).c_str());
}
static const char* cachePolicyToString(Cache::IOPolicy cachePolicy) {
  switch(cachePolicy) {
  case Cache::IOPolicy::fill:
    return "fill";
  case Cache::IOPolicy::flush:
    return "flush";
  case Cache::IOPolicy::local:
    return "local";
  case Cache::IOPolicy::bpfill:
    return "bpfill";
  case Cache::IOPolicy::epflush:
    return "epflush";
  case Cache::IOPolicy::fill_and_flush:
    return "fill_and_flush";
  case Cache::IOPolicy::unknown:
    return "unknown";
  }
  dawn_unreachable(
      std::string("invalid cache io policy" + std::to_string((unsigned int)cachePolicy)).c_str());
}

Cache::Cache(TypeKind type, IOPolicy policy, int fieldAccessID,
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
  return (policy_ != IOPolicy::local) || (type_ == TypeKind::bypass);
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

Cache::TypeKind Cache::getType() const { return type_; }

std::string Cache::getTypeAsString() const {
  switch(type_) {
  case TypeKind::IJ:
    return "cache_type::ij";
  case TypeKind::K:
    return "cache_type::k";
  case TypeKind::IJK:
    return "cache_type::ijk";
  case TypeKind::bypass:
    return "cache_type::bypass";
  }
  dawn_unreachable("invalid cache type");
}

Cache::IOPolicy Cache::getIOPolicy() const { return policy_; }

bool Cache::requiresWindow() const {
  return getIOPolicy() == IOPolicy::bpfill || getIOPolicy() == IOPolicy::epflush;
}

std::string Cache::getIOPolicyAsString() const {
  switch(policy_) {
  case IOPolicy::fill_and_flush:
    return "fill_and_flush";
  case IOPolicy::fill:
    return "fill";
  case IOPolicy::flush:
    return "flush";
  case IOPolicy::epflush:
    return "epflush";
  case IOPolicy::bpfill:
    return "bpfill";
  case IOPolicy::local:
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
