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

#ifndef DAWN_OPTIMIZER_FIELDACCESSEXTENTS_H
#define DAWN_OPTIMIZER_FIELDACCESSEXTENTS_H

#include "dawn/Optimizer/Extents.h"

namespace dawn {

class FieldAccessExtents {

public:
  FieldAccessExtents(Extents const& readExtents, Extents const& writeExtents)
      : readAccessExtents_(readExtents), writeAccessExtents_(writeExtents) {
    updateTotalExtents();
  }

  FieldAccessExtents(FieldAccessExtents&&) = default;
  FieldAccessExtents(FieldAccessExtents const&) = default;

  Extents const& getReadExtents() const { return readAccessExtents_; }
  Extents const& getWriteExtents() const { return writeAccessExtents_; }
  Extents const& getExtents() const { return totalExtents_; }

  void mergeReadExtents(Extents const& extents);

  void mergeWriteExtents(Extents const& extents);

private:
  void updateTotalExtents();

  Extents readAccessExtents_;
  Extents writeAccessExtents_;
  Extents totalExtents_;
};

} // namespace dawn
#endif
