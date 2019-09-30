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
//     IIRAccessExprData
//===------------------------------------------------------------------------------------------===//

IIRAccessExprData::IIRAccessExprData(const IIRAccessExprData& other) {
  AccessID = other.AccessID ? boost::make_optional(*other.AccessID) : other.AccessID;
}

bool IIRAccessExprData::operator==(const IIRAccessExprData& rhs) {
  return AccessID == rhs.AccessID;
}
bool IIRAccessExprData::operator!=(const IIRAccessExprData& rhs) { return !(*this == rhs); }

std::unique_ptr<ast::AccessExprData> IIRAccessExprData::clone() const {
  return make_unique<IIRAccessExprData>(*this);
}

int getAccessIDFromExpr(const std::shared_ptr<Expr>& expr) {
  switch(expr->getKind()) {
  case ast::Expr::EK_FieldAccessExpr:
    return *std::dynamic_pointer_cast<FieldAccessExpr>(expr)->getData<IIRAccessExprData>().AccessID;
  case ast::Expr::EK_LiteralAccessExpr:
    return *std::dynamic_pointer_cast<LiteralAccessExpr>(expr)
                ->getData<IIRAccessExprData>()
                .AccessID;
  case ast::Expr::EK_VarAccessExpr:
    return *std::dynamic_pointer_cast<VarAccessExpr>(expr)->getData<IIRAccessExprData>().AccessID;
  default:
    throw std::runtime_error("Invalid Expr to get access id from");
  }
}

} // namespace iir
} // namespace dawn
