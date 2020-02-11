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

#include "dawn/AST/LocationType.h"
#include <optional>

namespace dawn {
namespace iir {

/// @brief A variable can be a scalar (we want to remove scalar variables to get to valid IIR),
/// or it can have the same horizontal dimensions as a field (IJ for cartesian, location types for
/// unstructured).
///
/// @ingroup optimizer
enum class LocalVariableType { Scalar, OnCells, OnEdges, OnVertices, OnIJ };

/// @brief Contains information about a local variable
///
/// @ingroup optimizer
class LocalVariableData {

  // Type of the variable. std::nullopt means not computed yet.
  std::optional<LocalVariableType> type_;

public:
  /// @brief returns whether the type is set or not
  bool isTypeSet() const { return type_.has_value(); }
  /// @brief returns whether the variable is scalar (must be called once the type has
  /// been computed by PassLocalVarType)
  bool isScalar() const;
  /// @brief returns the computed variable type (must be called once the type has
  /// been computed by PassLocalVarType)
  LocalVariableType getType() const;
  /// @brief returns the ast::LocationType that corresponds to the variable type in the unstructured
  /// case (must be called once the type has been computed by PassLocalVarType and for a
  /// non-scalar variable)
  ast::LocationType getLocationType() const;

  void setType(LocalVariableType type) { type_ = type; }
};

} // namespace iir
} // namespace dawn
