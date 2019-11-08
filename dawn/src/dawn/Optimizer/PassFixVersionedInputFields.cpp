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
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/NodeUpdateType.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"

#include <memory>

namespace dawn {

static std::shared_ptr<ast::Stmt>
createAssignmentStatement(int assignmentID, int assigneeID,
                          iir::StencilMetaInformation const& metadata) {
  auto fa_assignee =
      std::make_shared<ast::FieldAccessExpr>(metadata.getNameFromAccessID(assigneeID));
  auto fa_assignment =
      std::make_shared<ast::FieldAccessExpr>(metadata.getNameFromAccessID(assignmentID));
  auto assignmentExpression =
      std::make_shared<ast::AssignmentExpr>(fa_assignment, fa_assignee, "=");
  auto assignmentStmt = iir::makeExprStmt(assignmentExpression);

  // Add access IDs for the new access expressions
  fa_assignment->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(assignmentID);
  fa_assignee->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(assigneeID);

  return assignmentStmt;
}

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

  auto domethod = std::make_unique<iir::DoMethod>(interval, metadata);
  // TODO: This may not be needed...
  domethod->setID(si->nextUID());

  domethod->getAST().push_back(std::move(assignmentStmt));

  domethod->update(iir::NodeUpdateType::level);

  return domethod;
}

/// @brief Creates the stage in which assignment happens (where the temporary gets filled)
static std::unique_ptr<iir::Stage>
createAssignmentStage(int assignmentID, int assigneeID,
                      std::shared_ptr<iir::StencilInstantiation> const& si,
                      const iir::Interval& interval) {
  // TODO: Use the IIRBuilder instead of doing this manually...

  // Create the do-method for the stage
  auto domethod = createDoMethod(assignmentID, assigneeID, si, interval);

  // Create the stage and add the do method
  iir::StencilMetaInformation& metadata = si->getMetaData();
  auto assignmentStage = std::make_unique<iir::Stage>(metadata, si->nextUID());
  assignmentStage->setExtents(iir::Extents{});

  assignmentStage->addDoMethod(std::move(domethod));
  assignmentStage->update(iir::NodeUpdateType::level);

  return assignmentStage;
}

/// @brief Creates a multistage in which assignment happens (where the versioned field gets filled)
static std::unique_ptr<iir::MultiStage>
createAssignmentMultiStage(int assignmentID, std::shared_ptr<iir::StencilInstantiation> const& si,
                           const iir::Interval& interval) {
  iir::StencilMetaInformation& metadata = si->getMetaData();
  int assigneeID = metadata.getOriginalVersionOfAccessID(assignmentID);

  auto assignmentStage = createAssignmentStage(assignmentID, assigneeID, si, interval);

  // Create the multistage and add the assignment stage
  auto ms = std::make_unique<iir::MultiStage>(metadata, iir::LoopOrderKind::Parallel);
  ms->setID(si->nextUID());

  ms->insertChild(std::move(assignmentStage));
  ms->update(iir::NodeUpdateType::levelAndTreeAbove);

  return ms;
}

PassFixVersionedInputFields::PassFixVersionedInputFields(OptimizerContext& context)
    : Pass(context, "PassFixVersionedInputFields", true) {}

bool PassFixVersionedInputFields::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& stencil : stencilInstantiation->getStencils()) {
    // Inserting multistages below, so can't use IterateIIROver here
    for(auto msiter = stencil->childrenBegin(); msiter != stencil->childrenEnd(); ++msiter) {
      for(auto& [id, field] : stencil->getFields()) {
        if(stencilInstantiation->getMetaData().isMultiVersionedField(id)) {
          // For every multistage, get the read access interval
          auto multiInterval = (*msiter)->computeReadAccessInterval(id);
          if(multiInterval.empty())
            continue;

          // If it has at least one, we need to generate a multistage that fills
          // all read intervals from the original field during the execution of
          // the multistage. Currently, we create one for each interval, but
          // these could later be merged into a single multistage.
          for(auto interval : multiInterval.getIntervals()) {
            auto insertedMultistage =
                createAssignmentMultiStage(id, stencilInstantiation, interval);
            msiter = std::next(stencil->insertChild(msiter, std::move(insertedMultistage)));
          }
        }
      }
    }
  }
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*stencilInstantiation->getIIR())) {
    doMethod->update(iir::NodeUpdateType::levelAndTreeAbove);
  }
  return true;
}
} // namespace dawn