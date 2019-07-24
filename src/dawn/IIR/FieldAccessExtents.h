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

#ifndef DAWN_IIR_FIELDACCESSEXTENTS_H
#define DAWN_IIR_FIELDACCESSEXTENTS_H

#include "dawn/IIR/Extents.h"
#include "dawn/Support/Json.h"

namespace dawn {
namespace iir {

/// @brief class storing the extents of accesses of a field within a computation
class FieldAccessExtents {

public:
  /// @brief constructors and assignment
  /// @{
  FieldAccessExtents(boost::optional<Extents> const& readExtents,
                     boost::optional<Extents> const& writeExtents)
      : readAccessExtents_(readExtents),
        writeAccessExtents_(writeExtents), totalExtents_{0, 0, 0, 0, 0, 0} {
    updateTotalExtents();
  }

  FieldAccessExtents() = delete;
  FieldAccessExtents(FieldAccessExtents&&) = default;
  FieldAccessExtents(FieldAccessExtents const&) = default;
  FieldAccessExtents& operator=(FieldAccessExtents&&) = default;
  FieldAccessExtents& operator=(const FieldAccessExtents&) = default;
  /// @}

  /// @brief getters
  /// @{
  boost::optional<Extents> const& getReadExtents() const { return readAccessExtents_; }
  boost::optional<Extents> const& getWriteExtents() const { return writeAccessExtents_; }
  Extents const& getExtents() const { return totalExtents_; }
  /// @}
  /// @brief merge of extent with another (argument) extent
  /// @{
  void mergeReadExtents(Extents const& extents);
  void mergeWriteExtents(Extents const& extents);
  void mergeReadExtents(boost::optional<Extents> const& extents);
  void mergeWriteExtents(boost::optional<Extents> const& extents);
  /// @}
  /// @brief setters
  /// @{
  void setReadExtents(Extents const& extents);
  void setWriteExtents(Extents const& extents);
  /// @}
  ///
  json::json jsonDump() const;

private:
  /// @brief update the total extent from read/write extents
  void updateTotalExtents();

  boost::optional<Extents> readAccessExtents_;
  boost::optional<Extents> writeAccessExtents_;
  Extents totalExtents_;
};

} // namespace iir
} // namespace dawn
#endif
