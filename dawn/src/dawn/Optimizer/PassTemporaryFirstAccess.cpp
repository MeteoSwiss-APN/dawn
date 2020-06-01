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
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/IndexRange.h"
#include "dawn/Support/Logger.h"
#include <algorithm>
#include <set>
#include <sstream>
#include <stack>
#include <unordered_map>

namespace dawn {

namespace {

class UnusedFieldVisitor : public iir::ASTVisitorForwarding {
  int AccessID_;
  bool fieldIsUnused_;
  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;
  std::stack<std::shared_ptr<const iir::StencilFunctionInstantiation>> functionInstantiationStack_;

public:
  UnusedFieldVisitor(int AccessID, const std::shared_ptr<iir::StencilInstantiation>& instantiation)
      : AccessID_(AccessID), fieldIsUnused_(false), instantiation_(instantiation) {}

  std::shared_ptr<const iir::StencilFunctionInstantiation>
  getStencilFunctionInstantiation(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
    if(!functionInstantiationStack_.empty())
      return functionInstantiationStack_.top()->getStencilFunctionInstantiation(expr);
    return instantiation_->getMetaData().getStencilFunctionInstantiation(expr);
  }

  void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override {
    std::shared_ptr<const iir::StencilFunctionInstantiation> funCall =
        getStencilFunctionInstantiation(expr);

    functionInstantiationStack_.push(funCall);
    fieldIsUnused_ |= funCall->isFieldUnused(AccessID_);

    // Follow the AST of the stencil function, it maybe unused in a nested stencil function
    funCall->getAST()->accept(*this);

    // Visit arguments
    iir::ASTVisitorForwarding::visit(expr);
    functionInstantiationStack_.pop();
  }

  bool isFieldUnused() const { return fieldIsUnused_; }
};

} // anonymous namespace

PassTemporaryFirstAccess::PassTemporaryFirstAccess(OptimizerContext& context)
    : Pass(context, "PassTemporaryFirstAccess", true) {}

bool PassTemporaryFirstAccess::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    std::unordered_map<int, iir::Stencil::FieldInfo> fields = stencilPtr->getFields();
    std::set<int> temporaryFields;

    auto tempFields = makeRange(fields, [](std::pair<int, iir::Stencil::FieldInfo> const& p) {
      return p.second.IsTemporary;
    });
    for(const auto& tmpF : tempFields) {
      temporaryFields.insert(tmpF.second.field.getAccessID());
    }

    // {AccesID : (isFirstAccessWrite, Stmt)}
    std::unordered_map<int, std::pair<bool, std::shared_ptr<iir::Stmt>>> accessMap;

    for(const auto& stmt : iterateIIROverStmt(*stencilPtr)) {
      const auto& accesses = stmt->getData<iir::IIRStmtData>().CallerAccesses;

      for(const auto& writeAccess : accesses->getWriteAccesses())
        if(temporaryFields.count(writeAccess.first))
          accessMap.emplace(writeAccess.first, std::make_pair(true, stmt));

      for(const auto& readAccess : accesses->getReadAccesses())
        if(temporaryFields.count(readAccess.first))
          accessMap.emplace(readAccess.first, std::make_pair(false, stmt));
    }

    for(const auto& accessPair : accessMap) {

      // Is first access to the temporary a read?
      if(accessPair.second.first == false) {
        int AccessID = accessPair.first;
        const std::shared_ptr<iir::Stmt>& stmt = accessPair.second.second;

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

        std::stringstream ss;
        ss << "Access to uninitialized temporary storage '" << nameLocPair.first << "'";

        dawn::DiagnosticStack stack;
        // Report notes where the temporary is referenced
        for(const auto& loc : nameLocPair.second) {
          stack.emplace(std::make_tuple(nameLocPair.first, loc));
        }

        throw SemanticError(ss.str() + createDiagnosticStackTrace("referenced here", stack));
      }
    }
  }

  return true;
}

} // namespace dawn
