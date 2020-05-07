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

#include "PassSetStageLocationType.h"
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
#include "dawn/Support/Exception.h"
#include "dawn/Support/Logger.h"
#include <deque>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

namespace dawn {
namespace {

// Try to deduce the location type of the given statement:
//  - assignment: location type of the left hand side (variable or field access)
//  - variable declaration: location type of variable
//  - stencil function call statement: deduce recursively from first statement of callee
//  - if statement: deduce recursively from first statement (part of then + else blocks)
//
// It is not possible to deduce the location type for any other case, so an error will be triggered.
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
  } else if(const auto& loopStmt = std::dynamic_pointer_cast<iir::LoopStmt>(stmt)) {
    if(auto* chainDesc =
           dynamic_cast<const ast::ChainIterationDescr*>(loopStmt->getIterationDescrPtr())) {
      return chainDesc->getChain().front();
    } else {
      dawn_unreachable("unsupported loop descriptor!\n");
    }
  }
  throw SemanticError(std::string("Couldn't deduce location type for statement at line ") +
                      static_cast<std::string>(stmt->getSourceLocation()) + ".");
}

} // namespace

bool PassSetStageLocationType::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& stage : iterateIIROver<iir::Stage>(*stencilInstantiation->getIIR())) {
    iir::DoMethod& doMethod = stage->getSingleDoMethod();
    const auto& stmts = doMethod.getAST().getStatements();

    auto locType = deduceLocationType(stmts[0], stencilInstantiation->getMetaData());
    stage->setLocationType(locType);
  }

  return true;
}

} // namespace dawn
