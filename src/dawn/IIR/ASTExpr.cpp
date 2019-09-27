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

#include "dawn/IIR/ASTExpr.h"
#include <memory>

namespace dawn {
namespace iir {

//===------------------------------------------------------------------------------------------===//
//     IIRVarAccessExprData
//===------------------------------------------------------------------------------------------===//

IIRVarAccessExprData::IIRVarAccessExprData(const IIRVarAccessExprData& other) {
  AccessID = other.AccessID ? boost::make_optional(*other.AccessID) : other.AccessID;
}

bool IIRVarAccessExprData::operator==(const IIRVarAccessExprData& rhs) {
  return AccessID == rhs.AccessID;
}
bool IIRVarAccessExprData::operator!=(const IIRVarAccessExprData& rhs) { return !(*this == rhs); }

std::unique_ptr<ast::VarAccessExprData> IIRVarAccessExprData::clone() const {
  return make_unique<IIRVarAccessExprData>(*this);
}

//===------------------------------------------------------------------------------------------===//
//     IIRFieldAccessExprData
//===------------------------------------------------------------------------------------------===//

IIRFieldAccessExprData::IIRFieldAccessExprData(const IIRFieldAccessExprData& other) {
  AccessID = other.AccessID ? boost::make_optional(*other.AccessID) : other.AccessID;
}

bool IIRFieldAccessExprData::operator==(const IIRFieldAccessExprData& rhs) {
  return AccessID == rhs.AccessID;
}
bool IIRFieldAccessExprData::operator!=(const IIRFieldAccessExprData& rhs) {
  return !(*this == rhs);
}

std::unique_ptr<ast::FieldAccessExprData> IIRFieldAccessExprData::clone() const {
  return make_unique<IIRFieldAccessExprData>(*this);
}

} // namespace iir
} // namespace dawn
