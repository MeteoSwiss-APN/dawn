#include "GridTypeChecker.h"
#include "dawn/AST/GridType.h"
#include "dawn/AST/Offsets.h"
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/MultiStage.h"
#include <optional>

namespace dawn {
bool GridTypeChecker::checkGridTypeConsistency(const dawn::iir::IIR& iir) {
  for(const auto& mSPtr : iterateIIROver<iir::MultiStage>(iir)) {
    for(const auto& field : mSPtr->getFields()) {
      std::vector<std::optional<iir::Extents>> extents = {
          field.second.getReadExtents(), field.second.getWriteExtents(),
          field.second.getReadExtentsRB(), field.second.getWriteExtentsRB()};
      for(const auto& extent : extents) {
        if(extent && extent->horizontalExtent().hasType() &&
           extent->horizontalExtent().getType() != iir.getGridType()) {
          return false;
        }
      }
    }
  }

  GridTypeChecker::TypeCheckerImpl typeChecker(iir.getGridType());
  for(const auto& doMethodPtr : iterateIIROver<iir::DoMethod>(iir)) {
    const std::shared_ptr<iir::BlockStmt>& ast =
        std::make_shared<iir::BlockStmt>(doMethodPtr->getAST());
    ast->accept(typeChecker);
    if(!typeChecker.isConsistent()) {
      return false;
    }
  }
  return true;
}
bool GridTypeChecker::checkGridTypeConsistency(const dawn::SIR& sir) {
  GridTypeChecker::TypeCheckerImpl typeChecker(sir.GridType);

  // check type consistency of stencil functions
  for(auto const& stenFunIt : sir.StencilFunctions) {
    for(const auto& astIt : stenFunIt->Asts) {
      astIt->accept(typeChecker);
      if(!typeChecker.isConsistent()) {
        return false;
      }
    }
  }

  // check type consistency of stencils
  for(const auto& stencil : sir.Stencils) {
    DAWN_ASSERT(stencil);
    const auto& stencilAst = stencil->StencilDescAst;
    stencilAst->accept(typeChecker);
    if(!typeChecker.isConsistent()) {
      return false;
    }
  }

  return true;
}

void GridTypeChecker::TypeCheckerImpl::visit(const std::shared_ptr<iir::FieldAccessExpr>& stmt) {
  if(!typesConsistent_) {
    return;
  }

  const auto& hOffset = stmt->getOffset().horizontalOffset();
  if(!hOffset.hasType()) {
    return;
  }
  typesConsistent_ &= hOffset.getGridType() == prescribedType_;
}

} // namespace dawn