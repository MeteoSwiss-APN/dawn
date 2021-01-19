#include "IndirectionChecker.h"

namespace dawn {
void IndirectionChecker::IndirectionCheckerImpl::visit(
    const std::shared_ptr<ast::AssignmentExpr>& expr) {
  lhs_ = true;
  expr->getLeft()->accept(*this);
  lhs_ = false;
  expr->getRight()->accept(*this);
}

void IndirectionChecker::IndirectionCheckerImpl::visit(
    const std::shared_ptr<ast::FieldAccessExpr>& expr) {
  if(!indirectionsValid_) {
    return;
  }

  if(lhs_ && expr->getOffset().hasVerticalIndirection()) {
    // indirections on lhs (i.e. vertically indirected wriste) are prohibited
    indirectionsValid_ = false;
    return;
  }

  if(expr->getOffset().hasVerticalIndirection()) {
    // inner offset must be null offset
    indirectionsValid_ =
        expr->getOffset().getVerticalIndirectionField()->getOffset().verticalShift() == 0 &&
        !expr->getOffset().getVerticalIndirectionField()->getOffset().hasVerticalIndirection();
  }
}

IndirectionChecker::IndirectionResult IndirectionChecker::checkIndirections(const dawn::SIR& SIR) {
  for(auto const& stenFunIt : SIR.StencilFunctions) {
    for(const auto& ast : stenFunIt->Asts) {
      IndirectionChecker::IndirectionCheckerImpl checker;
      for(const auto& stmt : ast->getRoot()->getChildren()) {
        stmt->accept(checker);
        if(!checker.indirectionsAreValid()) {
          return {false, stmt->getSourceLocation()};
        }
      }
    }
  }

  for(const auto& stencil : SIR.Stencils) {
    DAWN_ASSERT(stencil);
    const auto& stencilAst = stencil->StencilDescAst;
    IndirectionChecker::IndirectionCheckerImpl checker;
    for(const auto& stmt : stencilAst->getRoot()->getChildren()) {
      stmt->accept(checker);
      if(!checker.indirectionsAreValid()) {
        return {false, stmt->getSourceLocation()};
      }
    }
  }

  return {true, SourceLocation()};
}

IndirectionChecker::IndirectionResult
IndirectionChecker::checkIndirections(const dawn::iir::IIR& IIR) {
  for(const auto& stmt : iterateIIROverStmt(IIR)) {
    IndirectionChecker::IndirectionCheckerImpl checker;
    stmt->accept(checker);
    if(!checker.indirectionsAreValid()) {
      return {false, stmt->getSourceLocation()};
    }
  }
  return {true, SourceLocation()};
}

} // namespace dawn