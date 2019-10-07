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
  AccessID = other.AccessID ? std::make_optional(*other.AccessID) : other.AccessID;
}

bool IIRAccessExprData::operator==(const IIRAccessExprData& rhs) const {
  return AccessID == rhs.AccessID;
}
bool IIRAccessExprData::operator!=(const IIRAccessExprData& rhs) const { return !(*this == rhs); }

std::unique_ptr<ast::AccessExprData> IIRAccessExprData::clone() const {
  return std::make_unique<IIRAccessExprData>(*this);
}

bool IIRAccessExprData::equals(AccessExprData const* other) const {
  IIRAccessExprData const* iirAccessExprDataOther;
  return other && (iirAccessExprDataOther = dynamic_cast<IIRAccessExprData const*>(other)) &&
         *this == *iirAccessExprDataOther;
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
