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

#include "PassSplitStageByLocationType.h"
#include "dawn/AST/LocationType.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Support/Unreachable.h"
#include <deque>
#include <iterator>
#include <memory>
#include <stdexcept>

namespace dawn {
namespace {
// class DeduceLocationType : public iir::ASTVisitor {
// private:
//   iir::StencilMetaInformation const& metaInformation_;
//   std::optional<ast::LocationType> result_;

// public:
//   DeduceLocationType(iir::StencilMetaInformation const& metaInformation)
//       : metaInformation_(metaInformation) {}

//   ast::LocationType operator()(const std::shared_ptr<iir::Stmt>& stmt) {
//     stmt->accept(*this);
//     DAWN_ASSERT_MSG(result_.has_value(), "Couldn't deduce location type.");
//     return *result_;
//   }

//   // TODO: consider IF case, function call

//   void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
//     result_ = metaInformation_.getDenseLocationTypeFromAccessID(iir::getAccessID(expr));
//   }
// };

ast::LocationType deduceLocationType(const std::shared_ptr<iir::Stmt>& stmt,
                                     iir::StencilMetaInformation const& metaInformation) {
  if(const auto& exprStmt = std::dynamic_pointer_cast<iir::ExprStmt>(stmt)) {
    if(const auto& assignmentExpr =
           std::dynamic_pointer_cast<iir::AssignmentExpr>(exprStmt->getExpr())) {
      if(const std::shared_ptr<iir::FieldAccessExpr> fieldAccessExpr =
             std::dynamic_pointer_cast<iir::FieldAccessExpr>(assignmentExpr->getLeft())) {

        return metaInformation.getDenseLocationTypeFromAccessID(iir::getAccessID(fieldAccessExpr));

      } else if(const std::shared_ptr<iir::VarAccessExpr> varAccessExpr =
                    std::dynamic_pointer_cast<iir::VarAccessExpr>(assignmentExpr->getLeft())) {

        return metaInformation.getLocalVariableDataFromAccessID(iir::getAccessID(varAccessExpr))
            .getLocationType();
      }
    } else if(const std::shared_ptr<iir::StencilFunCallExpr> stencilFunCallExpr =
                  std::dynamic_pointer_cast<iir::StencilFunCallExpr>(exprStmt->getExpr())) {
      const auto& fun = metaInformation.getStencilFunctionInstantiation(stencilFunCallExpr);
      DAWN_ASSERT(fun);
      const auto& ast = fun->getDoMethod()->getAST();
      DAWN_ASSERT(ast.getStatements().size() != 0);
      return deduceLocationType(ast.getStatements()[0], metaInformation);
    }
  } else if(const auto& varDeclStmt = std::dynamic_pointer_cast<iir::VarDeclStmt>(stmt)) {

    return metaInformation.getLocalVariableDataFromAccessID(iir::getAccessID(varDeclStmt))
        .getLocationType();
  } else if(const auto& ifStmt = std::dynamic_pointer_cast<iir::IfStmt>(stmt)) {
    if(ifStmt->getThenStmt()->getChildren().size() != 0) {
      return deduceLocationType(ifStmt->getThenStmt()->getChildren()[0], metaInformation);
    } else if(ifStmt->hasElse()) {
      return deduceLocationType(ifStmt->getElseStmt()->getChildren()[0], metaInformation);
    }
  }
  dawn_unreachable("Unsupported statement");
}

// TODO check if statements are supported (same as PassRemoveScalars)

} // namespace

bool PassSplitStageByLocationType::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& multiStage : iterateIIROver<iir::MultiStage>(*stencilInstantiation->getIIR())) {
    for(auto stageIt = multiStage->childrenBegin(); stageIt != multiStage->childrenEnd();) {
      iir::Stage& stage = (**stageIt);
      iir::DoMethod& doMethod = stage.getSingleDoMethod();
      // TODO: change all deques to vectors otherwise we'll have O(n^2)
      std::deque<int> splitterIndices;
      std::deque<ast::LocationType> locationTypes;
      for(std::size_t i = 0; i < doMethod.getAST().getStatements().size(); ++i) {
        // auto locType = DeduceLocationType(stencilInstantiation->getMetaData())(
        //     doMethod.getAST().getStatements()[i]);
        auto locType = deduceLocationType(doMethod.getAST().getStatements()[i],
                                          stencilInstantiation->getMetaData());
        locationTypes.push_back(locType);

        if(i == doMethod.getAST().getStatements().size() - 1) // TODO fix this pattern
          break;
        splitterIndices.push_back(i);
      }

      if(!splitterIndices.empty()) {
        auto newStages = stage.split(splitterIndices, std::move(locationTypes));
        stageIt = multiStage->childrenErase(stageIt);
        stageIt = multiStage->insertChildren(stageIt, std::make_move_iterator(newStages.begin()),
                                             std::make_move_iterator(newStages.end()));
        std::advance(stageIt, newStages.size());
      } else {
        // TODO set location type
        ++stageIt;
      }
    }
  }

  return true;
}

} // namespace dawn
