//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
// extents
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "dawn/Optimizer/PassFixVersionedInputFields.h"
#include "dawn-c/ErrorHandling.h"
#include "dawn/AST/ASTExpr.h"
#include "dawn/AST/ASTStmt.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/NodeUpdateType.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/Logging.h"

#include <memory>

namespace dawn {

/// @brief Creates the assignment statement
static std::shared_ptr<ast::Stmt>
createAssignmentStatement(int assignmentID, int assigneeID,
                          iir::StencilMetaInformation const& metadata) {
  auto assignee = std::make_shared<ast::FieldAccessExpr>(metadata.getNameFromAccessID(assigneeID));
  auto assignment =
      std::make_shared<ast::FieldAccessExpr>(metadata.getNameFromAccessID(assignmentID));
  auto assignmentExpression = std::make_shared<ast::AssignmentExpr>(assignment, assignee, "=");
  auto assignmentStmt = iir::makeExprStmt(assignmentExpression);

  // Add access IDs for the new access expressions
  assignment->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(assignmentID);
  assignee->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(assigneeID);

  return assignmentStmt;
}

/// @brief Creates the do-method
static std::unique_ptr<iir::DoMethod>
createDoMethod(int assignmentID, int assigneeID,
               std::shared_ptr<iir::StencilInstantiation> const& si,
               const iir::Interval& interval) {
  iir::StencilMetaInformation& metadata = si->getMetaData();

  // Create the assignment statement with the new and old variables
  auto assignmentStmt = createAssignmentStatement(assignmentID, assigneeID, metadata);

  iir::Accesses accesses;
  accesses.addWriteExtent(assignmentID, iir::Extents{});
  accesses.addReadExtent(assigneeID, iir::Extents{});
  assignmentStmt->getData<iir::IIRStmtData>().CallerAccesses = std::move(accesses);

  auto doMethod = std::make_unique<iir::DoMethod>(interval, metadata);

  doMethod->getAST().push_back(std::move(assignmentStmt));
  doMethod->update(iir::NodeUpdateType::level);

  return doMethod;
}

/// @brief Creates the stage in which assignment happens (where the temporary gets filled)
static std::unique_ptr<iir::Stage>
createAssignmentStage(int assignmentID, int assigneeID,
                      std::shared_ptr<iir::StencilInstantiation> const& si,
                      const iir::Interval& interval, const iir::Extents& extents) {
  // Create the do-method for the stage
  auto doMethod = createDoMethod(assignmentID, assigneeID, si, interval);

  // Create the stage and add the do method
  iir::StencilMetaInformation& metadata = si->getMetaData();
  auto assignmentStage = std::make_unique<iir::Stage>(metadata, si->nextUID());
  assignmentStage->setExtents(extents);
  assignmentStage->addDoMethod(std::move(doMethod));
  assignmentStage->update(iir::NodeUpdateType::level);

  return assignmentStage;
}

/// @brief Creates a multistage in which assignment happens (where the versioned field gets filled)
static std::unique_ptr<iir::MultiStage>
createAssignmentMultiStage(int assignmentID, std::shared_ptr<iir::StencilInstantiation> const& si,
                           const iir::Interval& interval, const iir::Extents& extents) {
  iir::StencilMetaInformation& metadata = si->getMetaData();
  int assigneeID = metadata.getOriginalVersionOfAccessID(assignmentID);

  auto assignmentStage = createAssignmentStage(assignmentID, assigneeID, si, interval, extents);

  // Create the multistage and add the assignment stage
  auto ms = std::make_unique<iir::MultiStage>(metadata, iir::LoopOrderKind::Parallel);
  ms->setID(si->nextUID());

  ms->insertChild(std::move(assignmentStage));
  ms->update(iir::NodeUpdateType::levelAndTreeAbove);

  return ms;
}

/// @brief Collects AccessIDs from versioned fields in the IIR.
struct CollectVersionedIDs : public iir::ASTVisitorForwarding {
  const iir::StencilMetaInformation& metadata_;
  std::set<int> versionedAccessIDs;

  CollectVersionedIDs(const iir::StencilMetaInformation& metadata) : metadata_(metadata) {}

  virtual void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    const int id = iir::getAccessID(expr);
    if(metadata_.isAccessIDAVersion(id)) {
      versionedAccessIDs.insert(id);
    }
  }
};

PassFixVersionedInputFields::PassFixVersionedInputFields(OptimizerContext& context)
    : Pass(context, "PassFixVersionedInputFields", true) {}

bool PassFixVersionedInputFields::run(
    std::shared_ptr<iir::StencilInstantiation> const& stencilInstantiation) {
  for(const auto& stencil : stencilInstantiation->getStencils()) {
    // Inserting multistages below, so can't use IterateIIROver here
    for(auto msiter = stencil->childrenBegin(); msiter != stencil->childrenEnd(); ++msiter) {
      auto& ms = *msiter;
      CollectVersionedIDs getter{stencilInstantiation->getMetaData()};
      for(auto& stmt : iterateIIROverStmt(*ms)) {
        stmt->accept(getter);
      }
      for(int id : getter.versionedAccessIDs) {
        auto extents = ms->getField(id).getExtents();

        if(!extents.isVerticalPointwise()) {
          DAWN_LOG(WARNING) << "Cannot resolve race conditions with vertical dependencies. This "
                               "will generate potentially dangerous code.";
        }

        // If it has a non-zero extent, we need to generate a multistage that
        // fills all read intervals from the original field during the execution
        // of the multistage. Currently, we create one for each interval, but
        // these could later be merged into a single multistage.
        const auto multiInterval = ms->computeReadAccessInterval(id);
        for(const auto& interval : multiInterval.getIntervals()) {
          auto insertedMultistage =
              createAssignmentMultiStage(id, stencilInstantiation, interval, extents);
          msiter = stencil->insertChild(msiter, std::move(insertedMultistage));
          // update the mss: #TODO: this is still a workaround since we don't have level-and below:
          for(const auto& doMethods : iterateIIROver<iir::DoMethod>(**msiter)) {
            doMethods->update(iir::NodeUpdateType::levelAndTreeAbove);
          }
          ++msiter;
        }
      }
    }
  }

  return true;
}

} // namespace dawn
