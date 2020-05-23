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

#include "dawn/AST/Offsets.h"
#include "dawn/IIR/Extents.h"
#include <functional>
#include <unordered_map>

namespace dawn {
namespace iir {

class StencilFunctionInstantiation;
class StencilMetaInformation;

/// @brief Read and write accesses of a statement
///
/// Accesses are either part of a `StencilInstantiation` or `StencilFunctionInstantiation`.
/// @ingroup optimizer
class Accesses {
  std::unordered_map<int, Extents> writeAccesses_;
  std::unordered_map<int, Extents> readAccesses_;

public:
  Accesses() = default;
  Accesses(const Accesses&) = default;
  Accesses(Accesses&&) = default;
  Accesses& operator=(const Accesses&) = default;
  Accesses& operator=(Accesses&&) = default;

  bool operator==(const Accesses& rhs) const;
  bool operator!=(const Accesses& rhs) const;

  /// @brief Merge read offset/extent to the field given by `AccessID`
  ///
  /// @see Extents::merge
  /// @{
  void mergeReadOffset(int AccessID, const ast::Offsets& offset);
  void mergeReadExtent(int AccessID, const Extents& extent);
  /// @}

  /// @brief Merge write offset/extent to the field given by `AccessID`
  ///
  /// @see Extents::merge
  /// @{
  void mergeWriteOffset(int AccessID, const ast::Offsets& offset);
  void mergeWriteExtent(int AccessID, const Extents& extent);
  /// @}

  /// @brief Add write xtent to the field given by `AccessID`
  /// @see Extents::add
  void addReadExtent(int AccessID, const Extents& extent);

  /// @brief Add write extent to the field given by `AccessID`
  /// @see Extents::add
  void addWriteExtent(int AccessID, const Extents& extent);

  bool hasReadAccess(int accessID) const;
  bool hasWriteAccess(int accessID) const;

  bool hasAccess(int accessID) const;

  /// @brief Get access of field (returns NullExtent if field does not exist)
  const Extents& getReadAccess(int AccessID) const;
  const Extents& getWriteAccess(int AccessID) const;

  /// @brief Get the accesses maps
  std::unordered_map<int, Extents>& getReadAccesses() { return readAccesses_; }
  const std::unordered_map<int, Extents>& getReadAccesses() const { return readAccesses_; }

  std::unordered_map<int, Extents>& getWriteAccesses() { return writeAccesses_; }
  const std::unordered_map<int, Extents>& getWriteAccesses() const { return writeAccesses_; }

  /// @brief Convert the accesses of a stencil or stencil-function instantiation to string
  /// @{
  std::string toString(std::function<std::string(int)>&& accessIDToStringFunction,
                       std::size_t initialIndent = 0) const;
  /// @}

  /// @brief Report the accesses of a stencil or stencil-function instantiation
  /// @{
  std::string reportAccesses(const StencilFunctionInstantiation* stencilFunc) const;
  std::string reportAccesses(const StencilMetaInformation& metadata) const;
  /// @}
};

} // namespace iir
} // namespace dawn
