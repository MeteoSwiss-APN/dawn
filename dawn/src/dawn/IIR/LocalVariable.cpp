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

#include "LocalVariable.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace iir {

bool LocalVariableData::isScalar() const {
  DAWN_ASSERT_MSG(type_.has_value(), "Must run PassLocalVarType first.");
  return *type_ == LocalVariableType::Scalar;
}
LocalVariableType LocalVariableData::getType() const {
  DAWN_ASSERT_MSG(type_.has_value(), "Must run PassLocalVarType first.");
  return *type_;
}

ast::LocationType LocalVariableData::getLocationType() const {
  DAWN_ASSERT_MSG(type_.has_value(), "Must run PassLocalVarType first.");
  DAWN_ASSERT_MSG(!isScalar(), "Variable must not be scalar.");
  DAWN_ASSERT_MSG(*type_ != LocalVariableType::OnIJ,
                  "Variable is defined on cartesian dimensions.");

  switch(*type_) {
  case LocalVariableType::OnCells:
    return ast::LocationType::Cells;
  case LocalVariableType::OnEdges:
    return ast::LocationType::Edges;
  case LocalVariableType::OnVertices:
    return ast::LocationType::Vertices;
  default:
    dawn_unreachable("It should be an unstructured horizontal dimension.");
  }
}

} // namespace iir
} // namespace dawn