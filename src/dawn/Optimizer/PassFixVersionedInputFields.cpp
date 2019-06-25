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
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"

namespace dawn {

static std::unique_ptr<iir::StatementAccessesPair>
makeAssignmentStatement(int assignmentID, int assigneeID, iir::StencilMetaInformation& metadata) {
  auto fa_assignee = std::make_shared<FieldAccessExpr>(metadata.getNameFromAccessID(assigneeID));
  auto fa_assignment =
      std::make_shared<FieldAccessExpr>(metadata.getNameFromAccessID(assignmentID));
  auto assignmentExpression = std::make_shared<AssignmentExpr>(fa_assignment, fa_assignee, "=");
  auto expAssignment = std::make_shared<ExprStmt>(assignmentExpression);
  auto assignmentStatement = std::make_shared<Statement>(expAssignment, nullptr);

  // Add the new expressions to the map
  metadata.insertExprToAccessID(fa_assignment, assignmentID);
  metadata.insertExprToAccessID(fa_assignee, assigneeID);

  return make_unique<iir::StatementAccessesPair>(assignmentStatement);
}

static void addAssignmentToDoMethod(std::unique_ptr<iir::DoMethod>& domethod, int assignmentID,
                                    int assigneeID, iir::StencilMetaInformation& metadata) {
  // Create the StatementAccessPair of the assignment with the new and old variables
  auto assignmentStmtAccessPair = makeAssignmentStatement(assignmentID, assigneeID, metadata);
  auto newAccess = std::make_shared<iir::Accesses>();

  newAccess->addWriteExtent(assignmentID, iir::Extents(Array3i{{0, 0, 0}}));
  newAccess->addReadExtent(assigneeID, iir::Extents(Array3i{{0, 0, 0}}));
  assignmentStmtAccessPair->setAccesses(newAccess);
  domethod->insertChild(std::move(assignmentStmtAccessPair));
}
/// @brief Creates the stage in which assignment happens (where the temporary gets filled)
static std::unique_ptr<iir::Stage> createAssignmentStage(const iir::Interval& interval,
                                                         int assignmentID, int assigneeID,
                                                         iir::StencilMetaInformation& metadata,
                                                         int newStageID) {
  // Add the stage that assined the assingee to the assignment
  std::unique_ptr<iir::Stage> assignmentStage =
      make_unique<iir::Stage>(metadata, newStageID, interval);
  iir::Stage::DoMethodSmartPtr_t domethod = make_unique<iir::DoMethod>(interval, metadata);
  domethod->clearChildren();

  addAssignmentToDoMethod(domethod, assignmentID, assigneeID, metadata);

  // Add the single do method to the new Stage
  assignmentStage->clearChildren();
  assignmentStage->addDoMethod(domethod);
  for(auto& doMethod : assignmentStage->getChildren()) {
    doMethod->update(iir::NodeUpdateType::level);
  }
  assignmentStage->update(iir::NodeUpdateType::level);
  return assignmentStage;
}

PassFixVersionedInputFields::PassFixVersionedInputFields() : Pass("PassSetStageName", true) {}

bool PassFixVersionedInputFields::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(int id : stencilInstantiation->getMetaData()
                   .getFieldAccessMetadata()
                   .getVariableVersions()
                   .getVersionIDs()) {
    for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
      iir::Stencil& stencil = *stencilPtr;
      for(const auto& mss : iterateIIROver<iir::MultiStage>(stencil)) {
        // For every mulitstage, check if the versioned field as a read-access first
        auto multiInterval = mss->computeReadAccessInterval(id);
        if(multiInterval.empty())
          continue;

        // If it has one, we need to generate a domethod that fills that access from the
        // original field in every interval that is being accessed before it's first write
        for(auto interval : multiInterval.getIntervals()) {

          // we create the new stage that holds these do-methods
          auto insertedStage = createAssignmentStage(
              interval, id, stencilInstantiation->getMetaData().getOriginalVersionOfAccessID(id),
              stencilInstantiation->getMetaData(), stencilInstantiation->nextUID());
          // and insert them at the beginnning of the MultiStage
          mss->insertChild(mss->childrenBegin(), std::move(insertedStage));
        }
        // update the mss: #TODO: this is still a workaround since we don't have level-and below:
        for(const auto& domethods : iterateIIROver<iir::DoMethod>(*mss)) {
          domethods->update(iir::NodeUpdateType::levelAndTreeAbove);
        }
      }
    }
  }
  return true;
}

} // namespace dawn
