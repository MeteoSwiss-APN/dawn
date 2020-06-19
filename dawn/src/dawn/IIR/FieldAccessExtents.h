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

#pragma once

#include "dawn/IIR/Extents.h"
#include "dawn/Support/Json.h"
#include <optional>

namespace dawn {
namespace iir {

/// @brief class storing the extents of accesses of a field within a computation
class FieldAccessExtents {

public:
  /// @brief constructors and assignment
  /// @{
  FieldAccessExtents(std::optional<Extents> const& readExtents,
                     std::optional<Extents> const& writeExtents)
      : readAccessExtents_(readExtents), writeAccessExtents_(writeExtents) {}
  /// @}

  /// @brief getters
  /// @{
  std::optional<Extents> const& getReadExtents() const { return readAccessExtents_; }
  std::optional<Extents> const& getWriteExtents() const { return writeAccessExtents_; }
  Extents getExtents() const;
  /// @}
  /// @brief merge of extent with another (argument) extent
  /// @{
  void mergeReadExtents(Extents const& extents);
  void mergeWriteExtents(Extents const& extents);
  void mergeReadExtents(std::optional<Extents> const& extents);
  void mergeWriteExtents(std::optional<Extents> const& extents);
  /// @}
  /// @brief setters
  /// @{
  void setReadExtents(Extents const& extents);
  void setWriteExtents(Extents const& extents);
  /// @}
  ///
  json::json jsonDump() const;

private:
  std::optional<Extents> readAccessExtents_;
  std::optional<Extents> writeAccessExtents_;
};

} // namespace iir
} // namespace dawn
