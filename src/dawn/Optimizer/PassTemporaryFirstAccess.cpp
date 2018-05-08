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

#include "dawn/Optimizer/PassTemporaryFirstAccess.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/ASTVisitor.h"
#include <algorithm>
#include <set>
#include <stack>
#include <unordered_map>

namespace dawn {

namespace {

class UnusedFieldVisitor : public ASTVisitorForwarding {
  int AccessID_;
  bool fieldIsUnused_;
  const std::shared_ptr<StencilInstantiation>& instantiation_;
  std::stack<std::shared_ptr<const StencilFunctionInstantiation>> functionInstantiationStack_;

public:
  UnusedFieldVisitor(int AccessID, const std::shared_ptr<StencilInstantiation>& instantiation)
      : AccessID_(AccessID), fieldIsUnused_(false), instantiation_(instantiation) {}

  std::shared_ptr<const StencilFunctionInstantiation>
  getStencilFunctionInstantiation(const std::shared_ptr<StencilFunCallExpr>& expr) {
    if(!functionInstantiationStack_.empty())
      return functionInstantiationStack_.top()->getStencilFunctionInstantiation(expr);
    return instantiation_->getStencilFunctionInstantiation(expr);
  }

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    std::shared_ptr<const StencilFunctionInstantiation> funCall =
        getStencilFunctionInstantiation(expr);

    functionInstantiationStack_.push(funCall);
    fieldIsUnused_ |= funCall->isFieldUnused(AccessID_);

    // Follow the AST of the stencil function, it maybe unused in a nested stencil function
    funCall->getAST()->accept(*this);

    // Visit arguments
    ASTVisitorForwarding::visit(expr);
    functionInstantiationStack_.pop();
  }

  bool isFieldUnused() const { return fieldIsUnused_; }
};

} // anonymous namespace

PassTemporaryFirstAccess::PassTemporaryFirstAccess() : Pass("PassTemporaryFirstAccess", true) {}

bool PassTemporaryFirstAccess::run(
    const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  for(auto& stencilPtr : stencilInstantiation->getStencils()) {
    std::vector<Stencil::FieldInfo> fields = stencilPtr->getFields();
    std::set<int> temporaryFields;

    for(int i = 0; i < fields.size(); ++i)
      if(fields[i].IsTemporary)
        temporaryFields.insert(stencilInstantiation->getAccessIDFromName(fields[i].Name));

    // {AccesID : (isFirstAccessWrite, Stmt)}
    std::unordered_map<int, std::pair<bool, std::shared_ptr<Stmt>>> accessMap;

    for(auto& multiStagePtr : stencilPtr->getMultiStages()) {
      for(auto& stagePtr : multiStagePtr->getStages()) {
        DoMethod& doMethod = stagePtr->getSingleDoMethod();

        for(int i = 0; i < doMethod.getStatementAccessesPairs().size(); ++i) {
          const auto& accesses = doMethod.getStatementAccessesPairs()[i]->getAccesses();
          const auto& astStatement =
              doMethod.getStatementAccessesPairs()[i]->getStatement()->ASTStmt;

          for(const auto& writeAccess : accesses->getWriteAccesses())
            if(temporaryFields.count(writeAccess.first))
              accessMap.emplace(writeAccess.first, std::make_pair(true, astStatement));

          for(const auto& readAccess : accesses->getReadAccesses())
            if(temporaryFields.count(readAccess.first))
              accessMap.emplace(readAccess.first, std::make_pair(false, astStatement));
        }
      }
    }

    for(const auto& accessPair : accessMap) {

      // Is first access to the temporary a read?
      if(accessPair.second.first == false) {
        int AccessID = accessPair.first;
        const std::shared_ptr<Stmt>& stmt = accessPair.second.second;

        // Check if the statment contains a stencil function call in which the field is unused.
        // If a field is unused, we still have to add it as an input field of the stencil function.
        // Thus, the uninitialized temporary anaylsis reports a false positive on this temporary
        // field!
        UnusedFieldVisitor visitor(AccessID, stencilInstantiation);
        stmt->accept(visitor);
        if(visitor.isFieldUnused())
          continue;

        // Report the error
        auto nameLocPair =
            stencilInstantiation->getOriginalNameAndLocationsFromAccessID(AccessID, stmt);
        DiagnosticsBuilder diagError(DiagnosticsKind::Error, nameLocPair.second[0]);

        diagError << "access to uninitialized temporary storage '" << nameLocPair.first << "'";
        context->getDiagnostics().report(diagError);

        // Report notes where the temporary is referenced
        for(int i = 1; i < nameLocPair.second.size(); ++i) {
          DiagnosticsBuilder diagNote(DiagnosticsKind::Note, nameLocPair.second[i]);
          diagNote << "'" << nameLocPair.first << "' referenced here";
          context->getDiagnostics().report(diagNote);
        }

        return false;
      }
    }
  }

  return true;
}

} // namespace dawn
