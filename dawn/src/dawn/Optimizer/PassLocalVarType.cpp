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

#include "PassLocalVarType.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"

namespace dawn {
namespace {

// Resets types of all variables to "not computed" (type_ = std::nullopt)
void resetVarTypes(iir::StencilMetaInformation& metadata) {
  for(const auto& pair : metadata.getAccessIDToLocalVariableDataMap()) {
    auto& data = metadata.getLocalVariableDataFromAccessID(pair.first);
    data = iir::LocalVariableData{};
  }
}

class VarTypeFinder : public ast::ASTVisitorForwarding {
  iir::StencilMetaInformation& metadata_;

  // Currently determined variable type.
  // Unset if visitor is not inside a `VarDeclStmt` or an `AssignmentExpr` to a local variable.
  std::optional<iir::LocalVariableType> curVarType_;

  void updateVariableType(int accessID, SourceLocation sourceLocation) {

    // Set the variable type (could still be changed when visiting another assignment)
    iir::LocalVariableData& data = metadata_.getLocalVariableDataFromAccessID(accessID);

    if(data.isTypeSet()) {
      iir::LocalVariableType previouslyComputedType = data.getType();

      if(previouslyComputedType == iir::LocalVariableType::Scalar) {
        data.setType(*curVarType_);
      } else if(previouslyComputedType != *curVarType_) {

        throw std::runtime_error(
            dawn::format("Invalid assignment to variable at line %d: rhs differs in location type",
                         sourceLocation.Line));
      }
    } else {
      data.setType(*curVarType_);
    }
  }

  void updateCurrentType(iir::LocalVariableType newType, SourceLocation sourceLocation) {
    if(curVarType_.has_value() && *curVarType_ != iir::LocalVariableType::Scalar &&
       *curVarType_ != newType) {

      throw std::runtime_error(
          dawn::format("Invalid assignment to variable at line %d: rhs differs in location type",
                       sourceLocation.Line));
    }
    curVarType_ = newType;
  }

public:
  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    if(curVarType_.has_value()) { // We are inside an assignment to variable / variable declaration
      iir::LocalVariableType newType;
      switch(metadata_.getDenseLocationTypeFromAccessID(iir::getAccessID(expr))) {
      case ast::LocationType::Cells:
        newType = iir::LocalVariableType::OnCells;
        break;
      case ast::LocationType::Edges:
        newType = iir::LocalVariableType::OnEdges;
        break;
      case ast::LocationType::Vertices:
        newType = iir::LocalVariableType::OnVertices;
        break;
      }

      updateCurrentType(newType, expr->getSourceLocation());
    }
  }
  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override {
    if(curVarType_.has_value()) { // We are inside an assignment / variable declaration
      const iir::LocalVariableData& data =
          metadata_.getLocalVariableDataFromAccessID(iir::getAccessID(expr));
      DAWN_ASSERT_MSG(data.isTypeSet(), "Variable accessed before being declared.");
      iir::LocalVariableType newType = data.getType();

      updateCurrentType(newType, expr->getSourceLocation());
    }
  }
  void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override {
    if(expr->getLeft()->getKind() == ast::Expr::Kind::VarAccessExpr) {

      if(curVarType_
             .has_value()) { // Variable assignment inside variable assignment, not supported.
        throw std::runtime_error(dawn::format(
            "Variable assignment inside rhs of variable assignment is not supported. Line %d.",
            expr->getSourceLocation().Line));
      }

      // Assume scalar if nothing proves the opposite during the visit
      curVarType_ = iir::LocalVariableType::Scalar;

      // Visit rhs
      expr->getRight()->accept(*this);

      updateVariableType(iir::getAccessID(expr->getLeft()), expr->getSourceLocation());

      // Unset varType_ as we are exiting the AssignmentExpr's visit
      curVarType_ = std::nullopt;
    }
  }
  void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override {

    // Assume scalar if nothing proves the opposite during the visit
    curVarType_ = iir::LocalVariableType::Scalar;

    // Visit rhs
    ast::ASTVisitorForwarding::visit(stmt);

    updateVariableType(iir::getAccessID(stmt), stmt->getSourceLocation());

    // Unset varType_ as we are exiting the VarDeclStmt's visit
    curVarType_ = std::nullopt;
  }
  VarTypeFinder(iir::StencilMetaInformation& metadata) : metadata_(metadata) {}
};
} // namespace

bool PassLocalVarType::run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  // Loop over all the statements of each stencil
  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {

    // As this might not be the first run of this pass, need to clear the variables' types in order
    // to recompute them
    resetVarTypes(stencilInstantiation->getMetaData());

    VarTypeFinder varTypeFinder(stencilInstantiation->getMetaData());
    for(const auto& stmt : iterateIIROverStmt(*stencilPtr)) {
      stmt->accept(varTypeFinder);
    }
  }
  return true;
}

} // namespace dawn
