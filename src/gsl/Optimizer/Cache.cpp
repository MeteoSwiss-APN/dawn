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

#include "gsl/Optimizer/Cache.h"
#include "gsl/Support/Unreachable.h"

namespace gsl {

Cache::Cache(CacheTypeKind type, CacheIOPolicy policy, int fieldAccessID)
    : type_(type), policy_(policy), AccessID_(fieldAccessID) {}

int Cache::getCachedFieldAccessID() const { return AccessID_; }

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
  gsl_unreachable("invalid cache type");
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
    gsl_unreachable("invalid cache type");
  }
}

} // namespace gsl
