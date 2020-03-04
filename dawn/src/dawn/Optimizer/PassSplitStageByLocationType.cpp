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
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include <deque>
#include <stdexcept>

namespace dawn {
namespace {
class DeduceLocationType : public iir::ASTVisitorDisabled {
private:
  iir::StencilMetaInformation const& metaInformation_;
  std::optional<ast::LocationType> result_;

public:
  DeduceLocationType(iir::StencilMetaInformation const& metaInformation)
      : metaInformation_(metaInformation) {}

  ast::LocationType operator()(const std::shared_ptr<iir::Stmt>& stmt) {
    stmt->accept(*this);
    if(!result_)
      throw std::runtime_error("Couldn't deduce location type.");
    return *result_;
  }

  void visit(const std::shared_ptr<iir::ExprStmt>& stmt) override {
    stmt->getExpr()->accept(*this);
  }

  void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override {
    expr->getLeft()->accept(*this);
  }

  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    result_ = metaInformation_.getDenseLocationTypeFromAccessID(expr->getID());
  }
};

} // namespace

bool PassSplitStageByLocationType::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& stencil : stencilInstantiation->getStencils()) {
    for(const auto& multiStage : stencil->getChildren()) {
      for(auto stageIt = multiStage->childrenBegin(); stageIt != multiStage->childrenEnd();) {
        iir::Stage& stage = (**stageIt);
        iir::DoMethod& doMethod = stage.getSingleDoMethod();

        std::deque<int> splitterIndices;
        for(std::size_t i = 0; i < doMethod.getAST().getStatements().size(); ++i) {
          auto locType = DeduceLocationType(stencilInstantiation->getMetaData())(
              doMethod.getAST().getStatements()[i]);

          if(i == doMethod.getAST().getStatements().size() - 1) // TODO fix this pattern
            break;
          splitterIndices.push_back(i);
        }

        if(!splitterIndices.empty()) {
          auto newStages = stage.split(splitterIndices);
          stageIt = multiStage->childrenErase(stageIt);
          multiStage->insertChildren(stageIt, std::make_move_iterator(newStages.begin()),
                                     std::make_move_iterator(newStages.end()));
        }
      }
    }
  }

  return true;
}

} // namespace dawn
