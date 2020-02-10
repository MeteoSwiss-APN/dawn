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
  // Type inferred from the conditional expression of the if statement we are currently in.
  // Unset if either
  // - we are not inside an if statement or
  // - no field is accessed in the conditional
  std::optional<iir::LocalVariableType> conditionalType_;
  // Variables read within the conditional expression of the if statement we are currently in.
  // If we are not inside an if statement it's empty.
  std::set<int> variablesAccessedInConditional_;

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

  // Construct a LocalVariableType from a field accessed within an assignment to / declaration of a
  // local variable.
  iir::LocalVariableType inferLocalVarTypeFromField(int fieldAccessID) {
    iir::LocalVariableType type;

    if(sir::dimension_isa<sir::CartesianFieldDimension>(
           metadata_.getFieldDimensions(fieldAccessID).getHorizontalFieldDimension())) {
      // Cartesian case
      type = iir::LocalVariableType::OnIJ;
    } else {
      // Unstructured case, convert from location type
      switch(metadata_.getDenseLocationTypeFromAccessID(fieldAccessID)) {
      case ast::LocationType::Cells:
        type = iir::LocalVariableType::OnCells;
        break;
      case ast::LocationType::Edges:
        type = iir::LocalVariableType::OnEdges;
        break;
      case ast::LocationType::Vertices:
        type = iir::LocalVariableType::OnVertices;
        break;
      }
    }
    return type;
  }
  // Records that to write into `destinationVariable` we need to access `accessedVariable`.
  // `sourceLocation` is where this happens.
  void recordVariablePair(int accessedVariable, int destinationVariable,
                          SourceLocation sourceLocation) {
    // No self-dependencies
    if(accessedVariable != destinationVariable) {

      // If the set of dependers for this variable is not there, it means the variable hasn't been
      // declared
      DAWN_ASSERT_MSG(dependencyMap_.count(accessedVariable),
                      "Variable accessed before being declared.");
      // Need to register a dependency. Pair to be added is from dependency (variable accessed in
      // the rhs) to dependent (lhs).
      dependencyMap_.at(accessedVariable).insert(destinationVariable);

      // Retrieve the type of the accessed variable
      const iir::LocalVariableData& data =
          metadata_.getLocalVariableDataFromAccessID(accessedVariable);
      DAWN_ASSERT_MSG(data.isTypeSet(), "Variable accessed before being declared.");
      iir::LocalVariableType newType = data.getType();

      // Update the type of the current variable with `newType`
      updateVariableType(destinationVariable, newType, sourceLocation);
    }
  }

public:
  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    if(curVarID_.has_value()) { // We are inside an assignment to variable / variable declaration
      iir::LocalVariableType newType = inferLocalVarTypeFromField(iir::getAccessID(expr));
      // Update the type of the current variable with `newType`
      updateVariableType(*curVarID_, newType, expr->getSourceLocation());
    }
  }
  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override {
    // We are inside an assignment / variable declaration and the current variable accessed is not a
    // global variable
    if(curVarID_.has_value() && expr->isLocal()) {
      int accessedVariableID = iir::getAccessID(expr);
      recordVariablePair(accessedVariableID, *curVarID_, expr->getSourceLocation());
    }
  }
  void visit(const std::shared_ptr<iir::IfStmt>& ifStmt) override {
    // If in the conditional expression we are accessing a field, it means that every statement in
    // the then and else blocks will have a read-dependency on such field.

    for(const auto& readAccess :
        ifStmt->getCondStmt()->getData<iir::IIRStmtData>().CallerAccesses->getReadAccesses()) {
      int accessID = readAccess.first;
      if(metadata_.isAccessType(iir::FieldAccessType::Field, accessID)) {
        if(conditionalType_.has_value()) {
          if(*conditionalType_ != inferLocalVarTypeFromField(accessID)) {
            throw std::runtime_error(dawn::format(
                "Invalid if-condition at line %d: accesses fields with different location types",
                ifStmt->getSourceLocation().Line));
          }
        }
        // Record the accessed field's location type
        conditionalType_ = inferLocalVarTypeFromField(accessID);
      } else if(metadata_.isAccessType(iir::FieldAccessType::LocalVariable, accessID)) {
        variablesAccessedInConditional_.insert(accessID);
      }
    }
    // If we encounter an assignment to a local variable inside the then and else blocks, we should
    // impose the recorded type.
    ast::ASTVisitorForwarding::visit(ifStmt);
    // Unset conditionalType_ as we are exiting the visit of the if statement
    conditionalType_ = std::nullopt;
    variablesAccessedInConditional_ = {};
  }
  void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override {
    // Run only when lhs is a local variable
    if(expr->getLeft()->getKind() == ast::Expr::Kind::VarAccessExpr &&
       std::dynamic_pointer_cast<iir::VarAccessExpr>(expr->getLeft())->isLocal()) {

      if(curVarID_.has_value()) {
        // Variable assignment inside variable assignment, not supported.
        throw std::runtime_error(dawn::format("Variable assignment inside rhs of variable "
                                              "assignment is not supported. Line %d.",
                                              expr->getSourceLocation().Line));
      }
      // Set curVarID_ to current variable being processed
      curVarID_ = iir::getAccessID(expr->getLeft());
      // If we are inside an if-statement and the if condition accesses a field, we already have a
      // variable type inferred. Otherwise, assume scalar as starting type. In any case we need
      // to do the visit.
      const iir::LocalVariableType startingType =
          conditionalType_.has_value() ? *conditionalType_ : iir::LocalVariableType::Scalar;
      updateVariableType(*curVarID_, startingType, expr->getSourceLocation());
      // If the set of dependers for this variable is not there, it means the variable hasn't been
      // declared
      DAWN_ASSERT_MSG(dependencyMap_.count(*curVarID_), "Variable accessed before being declared.");
      // If we are inside an if statement, for each variable accessed in the conditional of the if,
      // we need to record its access as if it happened in the rhs of `expr`.
      for(int accessedVariableID : variablesAccessedInConditional_) {
        recordVariablePair(accessedVariableID, *curVarID_, expr->getSourceLocation());
      }
      // Visit rhs
      expr->getRight()->accept(*this);
      // Propagate the type information to variables that depend on this one
      propagateVariableType(*curVarID_, expr->getSourceLocation());
      // Unset curVarID_ as we are exiting the AssignmentExpr's visit
      curVarID_ = std::nullopt;
    }
  }
  void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override {
    // Set curVarID_ to current variable being processed
    curVarID_ = iir::getAccessID(stmt);
    // If we are inside an if-statement and the if condition accesses a field, we already have a
    // variable type inferred. Otherwise, assume scalar as starting type. In any case we need
    // to do the visit.
    const iir::LocalVariableType startingType =
        conditionalType_.has_value() ? *conditionalType_ : iir::LocalVariableType::Scalar;
    updateVariableType(*curVarID_, startingType, stmt->getSourceLocation());
    // Setup empty set of dependers for this variable
    dependencyMap_.emplace(*curVarID_, std::set<int>{});
    // If we are inside an if statement, for each variable accessed in the conditional of the if,
    // we need to record its access as if it happened in the rhs of `stmt`.
    for(int accessedVariableID : variablesAccessedInConditional_) {
      recordVariablePair(accessedVariableID, *curVarID_, stmt->getSourceLocation());
    }
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
