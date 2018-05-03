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

#include "dawn/Optimizer/FieldAccessExtents.h"

namespace dawn {
void FieldAccessExtents::mergeReadExtents(Extents const& extents) {
  readAccessExtents_.merge(extents);
  updateTotalExtents();
}
void FieldAccessExtents::mergeWriteExtents(Extents const& extents) {
  writeAccessExtents_.merge(extents);
  updateTotalExtents();
}

void FieldAccessExtents::updateTotalExtents() {
  totalExtents_ = readAccessExtents_;
  totalExtents_.merge(writeAccessExtents_);
}

} // namespace dawn
