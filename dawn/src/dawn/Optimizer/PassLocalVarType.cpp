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

  // AccessID of variable being processed.
  // Unset if visitor is not inside a `VarDeclStmt` or an `AssignmentExpr` to a local variable.
  std::optional<int> curVarID_;
  // When assigning to or declaring variable A, each other variable that appears on the rhs is a
  // dependence of A. This map is from a variable B to all variables that depend on B. A variable
  // cannot depend on itself.
  std::unordered_map<int, std::vector<int>> dependencyMap_;

  void updateVariableType(iir::LocalVariableType newType, SourceLocation sourceLocation) {
    DAWN_ASSERT(curVarID_.has_value());

    // Set the variable type (could still be changed when visiting another assignment)
    iir::LocalVariableData& data = metadata_.getLocalVariableDataFromAccessID(*curVarID_);

    if(data.isTypeSet()) {
      iir::LocalVariableType previouslyComputedType = data.getType();

      if(previouslyComputedType == iir::LocalVariableType::Scalar) {
        data.setType(newType);
      } else if(previouslyComputedType != newType) {

        throw std::runtime_error(
            dawn::format("Invalid assignment to variable at line %d: rhs differs in location type",
                         sourceLocation.Line));
      }
    } else {
      data.setType(newType);
    }
  }

public:
  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    if(curVarID_.has_value()) { // We are inside an assignment to variable / variable declaration
      iir::LocalVariableType newType;

      if(sir::dimension_isa<sir::CartesianFieldDimension>(
             metadata_.getFieldDimensions(iir::getAccessID(expr)).getHorizontalFieldDimension())) {
        // Cartesian case
        newType = iir::LocalVariableType::OnIJ;
      } else {
        // Unstructured case
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
      }

      updateVariableType(newType, expr->getSourceLocation());
    }
  }
  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override {
    if(curVarID_.has_value()) { // We are inside an assignment / variable declaration

      // Need to map dependency between lhs variable and accessed variable
      // TODO...

      // Retrieve the type of the accessed variable
      const iir::LocalVariableData& data =
          metadata_.getLocalVariableDataFromAccessID(iir::getAccessID(expr));
      DAWN_ASSERT_MSG(data.isTypeSet(), "Variable accessed before being declared.");
      iir::LocalVariableType newType = data.getType();

      updateVariableType(newType, expr->getSourceLocation());
    }
  }
  void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override {
    if(expr->getLeft()->getKind() == ast::Expr::Kind::VarAccessExpr) {

      if(curVarID_.has_value()) { // Variable assignment inside variable assignment, not supported.
        throw std::runtime_error(dawn::format(
            "Variable assignment inside rhs of variable assignment is not supported. Line %d.",
            expr->getSourceLocation().Line));
      }

      // Assume scalar if nothing proves the opposite during the visit
      curVarID_ = iir::getAccessID(expr->getLeft());
      updateVariableType(iir::LocalVariableType::Scalar, expr->getSourceLocation());

      // Visit rhs
      expr->getRight()->accept(*this);

      // Unset curVarID_ as we are exiting the AssignmentExpr's visit
      curVarID_ = std::nullopt;
    }
  }
  void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override {

    // Assume scalar if nothing proves the opposite during the visit
    curVarID_ = iir::getAccessID(stmt);
    updateVariableType(iir::LocalVariableType::Scalar, stmt->getSourceLocation());

    // Visit rhs
    ast::ASTVisitorForwarding::visit(stmt);

    // Unset curVarID_ as we are exiting the VarDeclStmt's visit
    curVarID_ = std::nullopt;
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
