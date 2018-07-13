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

/// @brief class storing the extents of accesses of a field within a computation
class FieldAccessExtents {

public:
  FieldAccessExtents(boost::optional<Extents> const& readExtents,
                     boost::optional<Extents> const& writeExtents)
      : readAccessExtents_(readExtents), writeAccessExtents_(writeExtents),
        totalExtents_{0, 0, 0, 0, 0, 0} {
    updateTotalExtents();
  }

  FieldAccessExtents() = delete;

  FieldAccessExtents(FieldAccessExtents&&) = default;
  FieldAccessExtents(FieldAccessExtents const&) = default;

  boost::optional<Extents> const& getReadExtents() const { return readAccessExtents_; }
  boost::optional<Extents> const& getWriteExtents() const { return writeAccessExtents_; }
  Extents const& getExtents() const { return totalExtents_; }

  void mergeReadExtents(Extents const& extents);
  void mergeWriteExtents(Extents const& extents);
  void mergeReadExtents(boost::optional<Extents> const& extents);
  void mergeWriteExtents(boost::optional<Extents> const& extents);

private:
  void updateTotalExtents();

  boost::optional<Extents> readAccessExtents_;
  boost::optional<Extents> writeAccessExtents_;
  Extents totalExtents_;
};

} // namespace dawn
#endif
