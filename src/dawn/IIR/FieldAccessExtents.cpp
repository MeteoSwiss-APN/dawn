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

namespace dawn {
namespace iir {
void FieldAccessExtents::mergeReadExtents(Extents const& extents) {
  if(readAccessExtents_.is_initialized())
    readAccessExtents_->merge(extents);
  else
    readAccessExtents_ = boost::make_optional(extents);
  updateTotalExtents();
}
void FieldAccessExtents::mergeWriteExtents(Extents const& extents) {
  if(writeAccessExtents_.is_initialized())
    writeAccessExtents_->merge(extents);
  else
    writeAccessExtents_ = boost::make_optional(extents);

  updateTotalExtents();
}

void FieldAccessExtents::mergeReadExtents(boost::optional<Extents> const& extents) {
  if(extents.is_initialized())
    mergeReadExtents(*extents);
}
void FieldAccessExtents::mergeWriteExtents(boost::optional<Extents> const& extents) {
  if(extents.is_initialized())
    mergeWriteExtents(*extents);
}

void FieldAccessExtents::updateTotalExtents() {
  if(readAccessExtents_.is_initialized()) {
    totalExtents_ = *readAccessExtents_;
    if(writeAccessExtents_.is_initialized())
      totalExtents_.merge(*writeAccessExtents_);
  } else if(writeAccessExtents_.is_initialized()) {
    totalExtents_ = *writeAccessExtents_;
  }
}

} // namespace iir
} // namespace dawn
