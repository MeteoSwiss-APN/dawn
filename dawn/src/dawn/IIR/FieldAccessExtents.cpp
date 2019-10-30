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
#include "dawn/IIR/FieldAccessExtents.h"
#include "dawn/Support/Json.h"
#include <sstream>

namespace dawn {
namespace iir {
void FieldAccessExtents::mergeReadExtents(Extents const& extents) {
  if(readAccessExtents_)
    readAccessExtents_->merge(extents);
  else
    readAccessExtents_ = std::make_optional(extents);
}
json::json FieldAccessExtents::jsonDump() const {
  json::json node;
  std::stringstream ss;
  if(readAccessExtents_) {
    ss << *readAccessExtents_;
  } else {
    ss << "null";
  }

  node["read_access"] = ss.str();
  ss.str("");
  if(writeAccessExtents_) {
    ss << *writeAccessExtents_;
  } else {
    ss << "null";
  }

  node["write_access"] = ss.str();
  return node;
}

void FieldAccessExtents::mergeWriteExtents(Extents const& extents) {
  if(writeAccessExtents_)
    writeAccessExtents_->merge(extents);
  else
    writeAccessExtents_ = std::make_optional(extents);
}
void FieldAccessExtents::mergeReadExtents(std::optional<Extents> const& extents) {
  if(extents)
    mergeReadExtents(*extents);
}
void FieldAccessExtents::mergeWriteExtents(std::optional<Extents> const& extents) {
  if(extents)
    mergeWriteExtents(*extents);
}

void FieldAccessExtents::setReadExtents(Extents const& extents) {
  readAccessExtents_ = std::make_optional(extents);
}
void FieldAccessExtents::setWriteExtents(Extents const& extents) {
  writeAccessExtents_ = std::make_optional(extents);
}

Extents FieldAccessExtents::getExtents() const {
  if(readAccessExtents_ && writeAccessExtents_) {
    auto totalExtents = *readAccessExtents_;
    totalExtents.merge(*writeAccessExtents_);
    return totalExtents;
  } else if(readAccessExtents_) {
    return *readAccessExtents_;
  } else if(writeAccessExtents_) {
    return *writeAccessExtents_;
  } else {
    DAWN_ASSERT("no extents available");
    dawn_unreachable("");
  }
}

} // namespace iir
} // namespace dawn
