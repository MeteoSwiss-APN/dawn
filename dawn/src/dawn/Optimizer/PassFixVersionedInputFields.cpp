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

#include "dawn/Optimizer/PassFixVersionedInputFields.h"
#include "dawn/AST/ASTExpr.h"
#include "dawn/AST/ASTStmt.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"

#include <memory>

namespace dawn {

static std::shared_ptr<ast::Stmt> makeAssignmentStatement(int assignmentID, int assigneeID,
                                                          iir::StencilMetaInformation& metadata) {
  auto fa_assignee =
      std::make_shared<ast::FieldAccessExpr>(metadata.getNameFromAccessID(assigneeID));
  auto fa_assignment =
      std::make_shared<ast::FieldAccessExpr>(metadata.getNameFromAccessID(assignmentID));
  auto assignmentExpression =
      std::make_shared<ast::AssignmentExpr>(fa_assignment, fa_assignee, "=");
  auto expAssignment = iir::makeExprStmt(assignmentExpression);

  // Add access IDs for the new access expressions
  fa_assignment->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(assignmentID);
  fa_assignee->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(assigneeID);

  return expAssignment;
}

static void addAssignmentToDoMethod(std::unique_ptr<iir::DoMethod>& domethod, int assignmentID,
                                    int assigneeID, iir::StencilMetaInformation& metadata) {
  // Create the assignment statement with the new and old variables
  auto assignmentStmt = makeAssignmentStatement(assignmentID, assigneeID, metadata);

  iir::Accesses accesses;
  accesses.addWriteExtent(assignmentID, iir::Extents{});
  accesses.addReadExtent(assigneeID, iir::Extents{});

  assignmentStmt->getData<iir::IIRStmtData>().CallerAccesses = std::move(accesses);
  domethod->getAST().push_back(std::move(assignmentStmt));
}

/// @brief Creates the stage in which assignment happens (where the temporary gets filled)
static std::unique_ptr<iir::Stage> createAssignmentStage(const iir::Interval& interval,
                                                         int assignmentID, int assigneeID,
                                                         iir::StencilMetaInformation& metadata,
                                                         int newStageID) {
  // Add the stage that assigned the assignee to the assignment
  auto assignmentStage = std::make_unique<iir::Stage>(metadata, newStageID, interval);
  auto domethod = std::make_unique<iir::DoMethod>(interval, metadata);
  // domethod->clearChildren();

  addAssignmentToDoMethod(domethod, assignmentID, assigneeID, metadata);

  // Add the single do method to the new Stage
  assignmentStage->clearChildren();
  assignmentStage->addDoMethod(std::move(domethod));
  for(auto& doMethod : assignmentStage->getChildren()) {
    doMethod->update(iir::NodeUpdateType::level);
  }
  assignmentStage->update(iir::NodeUpdateType::level);
  return assignmentStage;
}

PassFixVersionedInputFields::PassFixVersionedInputFields(OptimizerContext& context)
    : Pass(context, "PassFixVersionedInputFields", true) {}

bool PassFixVersionedInputFields::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;
    for(auto& [id, field] : stencil.getFields()) {
      if(stencilInstantiation->getMetaData().isMultiVersionedField(id)) {
        for(const auto& mss : iterateIIROver<iir::MultiStage>(stencil)) {
          // For every multistage, check if the versioned field is a read-access first
          auto multiInterval = mss->computeReadAccessInterval(id);
          if(multiInterval.empty())
            continue;

          // If it has one, we need to generate a do-method that fills that access from the
          // original field in every interval that is being accessed before its first write
          for(auto interval : multiInterval.getIntervals()) {

            // we create the new stage that holds these do-methods
            auto insertedStage = createAssignmentStage(
                interval, id, stencilInstantiation->getMetaData().getOriginalVersionOfAccessID(id),
                stencilInstantiation->getMetaData(), stencilInstantiation->nextUID());
            // and insert them at the beginnning of the MultiStage
            std::cout << "calling insertChild" << std::endl;
            mss->insertChild(mss->childrenBegin(), std::move(insertedStage));
          }
          // update the mss: #TODO: this is still a workaround since we don't have level-and below:
          for(const auto& domethods : iterateIIROver<iir::DoMethod>(*mss)) {
            domethods->update(iir::NodeUpdateType::levelAndTreeAbove);
          }
        }
      }
    }
  }
  return true;
}

} // namespace dawn