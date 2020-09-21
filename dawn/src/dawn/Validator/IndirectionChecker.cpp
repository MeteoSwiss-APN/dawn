#include "IndirectionChecker.h"

namespace dawn {
void IndirectionChecker::IndirectionCheckerImpl::visit(
    const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  if(!indirectionsValid_) {
    return;
  }

  auto indirection = expr->getOffset().verticalIndirectionAsField();
  if(indirection.has_value()) {
    // inner offset must be null offset
    indirectionsValid_ = indirection.value()->getOffset().verticalOffset() == 0 &&
                         !indirection.value()->getOffset().verticalIndirectionAsField().has_value();
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
  for(const auto& stage : iterateIIROver<iir::Stage>(IIR)) {
    for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*stage)) {
      for(const auto& stmt : doMethod->getAST().getStatements()) {
        IndirectionChecker::IndirectionCheckerImpl checker;
        stmt->accept(checker);
        if(!checker.indirectionsAreValid()) {
          return {false, stmt->getSourceLocation()};
        }
      }
    }
  }
  return {true, SourceLocation()};
}

} // namespace dawn