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

class VarTypeFinder : public ast::ASTVisitorForwarding {

  iir::StencilMetaInformation& metadata_;
  // AccessID of variable being processed.
  // Unset if visitor is not inside a `VarDeclStmt` or an `AssignmentExpr` to a local variable.
  std::optional<int> curVarID_;
  // When assigning to or declaring variable A, each (other) variable that appears on the rhs is a
  // dependence of A. This map is from a variable B to all variables that depend on B. A variable
  // cannot depend on itself.
  std::unordered_map<int, std::set<int>> dependencyMap_;

  // Sets the type of variable with id `varID` to `newType` if the previous type was scalar or
  // unset. Throws if non-scalar previous type is different from `newType` (mixing different
  // location types).
  void updateVariableType(int varID, iir::LocalVariableType newType,
                          SourceLocation sourceLocation) {

    iir::LocalVariableData& data = metadata_.getLocalVariableDataFromAccessID(varID);
    if(data.isTypeSet()) {
      if(newType != iir::LocalVariableType::Scalar) {
        iir::LocalVariableType previouslyComputedType = data.getType();

        if(previouslyComputedType == iir::LocalVariableType::Scalar) {
          data.setType(newType);
        } else if(previouslyComputedType != newType) {
          throw std::runtime_error(dawn::format(
              "Invalid assignment to variable at line %d: rhs differs in location type",
              sourceLocation.Line));
        }
      }
    } else {
      data.setType(newType);
    }
  }
  // Propagates the variable type of variable with id `accessID` to all variables that depend on it.
  // `sourceLocation` is for reporting the line number in case an illegal assignment is detected.
  void propagateVariableType(int accessID, SourceLocation sourceLocation) {
    // Set of visited variables (to avoid loops)
    std::set<int> visitedSet;
    // Define a recursive lambda function to propagate the type through the dependency map
    std::function<void(int)> recursivePropagate;
    recursivePropagate = [&](int dependee) {
      // Flag the variable as visited
      visitedSet.insert(dependee);
      // If no other variable depends on the dependee, return
      if(!dependencyMap_.count(dependee)) {
        return;
      }
      // Collect the type to propagate
      const iir::LocalVariableData& dependeeData =
          metadata_.getLocalVariableDataFromAccessID(dependee);
      // If the dependee's type is not set yet, it has been accessed before being declared.
      DAWN_ASSERT_MSG(dependeeData.isTypeSet(), "Variable accessed before being declared.");
      // Loop through variables (dependers) which depend on the dependee
      for(int depender : dependencyMap_.at(dependee)) {
        // Make sure depender hasn't be already visited within the current call to
        // `propagateVariableType()`
        if(!visitedSet.count(depender)) {
          // Update the type of depender with the dependee's type
          updateVariableType(depender, dependeeData.getType(), sourceLocation);
          // Call the recursive procedure on depender
          recursivePropagate(depender);
        }
      }
    };
    // Call the recursive function
    recursivePropagate(accessID);
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
        // Unstructured case, convert to location type
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
      // Update the type of the current variable with `newType`
      updateVariableType(*curVarID_, newType, expr->getSourceLocation());
    }
  }
  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override {
    // We are inside an assignment / variable declaration and the current variable accessed is not a
    // global variable
    if(curVarID_.has_value() && expr->isLocal()) {
      int accessedVariableID = iir::getAccessID(expr);
      // No self-dependencies
      if(accessedVariableID != *curVarID_) {

        // If the set of dependers for this variable is not there, it means the variable hasn't been
        // declared
        DAWN_ASSERT_MSG(dependencyMap_.count(accessedVariableID),
                        "Variable accessed before being declared.");
        // Need to register a dependency. Pair to be added is from dependency (variable accessed in
        // the rhs) to dependent (lhs).
        dependencyMap_.at(accessedVariableID).insert(*curVarID_);

        // Retrieve the type of the accessed variable
        const iir::LocalVariableData& data =
            metadata_.getLocalVariableDataFromAccessID(accessedVariableID);
        DAWN_ASSERT_MSG(data.isTypeSet(), "Variable accessed before being declared.");
        iir::LocalVariableType newType = data.getType();

        // Update the type of the current variable with `newType`
        updateVariableType(*curVarID_, newType, expr->getSourceLocation());
      }
    }
  }
  void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override {
    // Run only when lhs is a local variable
    if(expr->getLeft()->getKind() == ast::Expr::Kind::VarAccessExpr &&
       std::dynamic_pointer_cast<iir::VarAccessExpr>(expr->getLeft())->isLocal()) {

      if(curVarID_.has_value()) { // Variable assignment inside variable assignment, not supported.
        throw std::runtime_error(dawn::format(
            "Variable assignment inside rhs of variable assignment is not supported. Line %d.",
            expr->getSourceLocation().Line));
      }
      // Assume scalar if nothing proves the opposite during the visit
      curVarID_ = iir::getAccessID(expr->getLeft());
      updateVariableType(*curVarID_, iir::LocalVariableType::Scalar, expr->getSourceLocation());
      // If the set of dependers for this variable is not there, it means the variable hasn't been
      // declared
      DAWN_ASSERT_MSG(dependencyMap_.count(*curVarID_), "Variable accessed before being declared.");
      // Visit rhs
      expr->getRight()->accept(*this);
      // Propagate the type information to variables that depend on this one
      propagateVariableType(*curVarID_, expr->getSourceLocation());
      // Unset curVarID_ as we are exiting the AssignmentExpr's visit
      curVarID_ = std::nullopt;
    }
  }
  void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override {
    // Assume scalar if nothing proves the opposite during the visit
    curVarID_ = iir::getAccessID(stmt);
    updateVariableType(*curVarID_, iir::LocalVariableType::Scalar, stmt->getSourceLocation());
    // Setup empty set of dependers for this variable
    dependencyMap_.emplace(*curVarID_, std::set<int>{});
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
    stencilInstantiation->getMetaData().resetLocalVarTypes();

    for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*stencilPtr)) {
      // Local variables are local to a DoMethod. VarTypeFinder needs to be applied to each DoMethod
      // separately.
      VarTypeFinder varTypeFinder(stencilInstantiation->getMetaData());
      doMethod->getAST().accept(varTypeFinder);
    }
  }
  return true;
}

} // namespace dawn
