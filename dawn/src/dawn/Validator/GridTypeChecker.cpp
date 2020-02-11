#include "GridTypeChecker.h"
#include "dawn/AST/GridType.h"
#include "dawn/AST/Offsets.h"
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/SIR/SIR.h"
#include <optional>

namespace dawn {
bool GridTypeChecker::checkGridTypeConsistency(const dawn::iir::IIR& iir) {
  // Check LocalVariableDatas
  for(const auto& stencil : iterateIIROver<iir::Stencil>(iir)) {
    for(const auto& pair : stencil->getMetadata().getAccessIDToLocalVariableDataMap()) {
      if(pair.second.isTypeSet()) {
        iir::LocalVariableType varType = pair.second.getType();
        switch(varType) {
        case iir::LocalVariableType::Scalar:
          continue;
        case iir::LocalVariableType::OnCells:
        case iir::LocalVariableType::OnEdges:
        case iir::LocalVariableType::OnVertices:
          if(iir.getGridType() == ast::GridType::Cartesian) {
            return false;
          }
          break;
        case iir::LocalVariableType::OnIJ:
          if(iir.getGridType() == ast::GridType::Unstructured) {
            return false;
          }
          break;
        }
      }
    }
  }

  for(const auto& mSPtr : iterateIIROver<iir::MultiStage>(iir)) {
    for(const auto& field : mSPtr->getFields()) {
      // Check Extents
      std::vector<std::optional<iir::Extents>> extents = {
          field.second.getReadExtents(), field.second.getWriteExtents(),
          field.second.getReadExtentsRB(), field.second.getWriteExtentsRB()};
      for(const auto& extent : extents) {
        if(extent && extent->horizontalExtent().hasType() &&
           extent->horizontalExtent().getType() != iir.getGridType()) {
          return false;
        }
      }

      // Check FieldDimensions
      const auto& hDimension = field.second.getFieldDimensions().getHorizontalFieldDimension();
      if(hDimension.getType() != iir.getGridType()) {
        return false;
      }
    }
  }

  GridTypeChecker::TypeCheckerImpl typeChecker(iir.getGridType());
  for(const auto& doMethodPtr : iterateIIROver<iir::DoMethod>(iir)) {
    const auto& ast = doMethodPtr->getASTPtr();
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
    for(auto arg : stenFunIt->Args) {
      if(arg->Kind == sir::StencilFunctionArg::ArgumentKind::Field) {
        sir::Field* field = (sir::Field*)arg.get();
        // Check FieldDimensions
        const auto& hDimension = field->Dimensions.getHorizontalFieldDimension();
        if(hDimension.getType() != sir.GridType) {
          return false;
        }
      }
    }
    for(const auto& astIt : stenFunIt->Asts) {
      astIt->accept(typeChecker);
      if(!typeChecker.isConsistent()) {
        return false;
      }
    }
  }

  // check type consistency of stencils
  for(const auto& stencil : sir.Stencils) {
    for(const auto& field : stencil->Fields) {
      // Check FieldDimensions
      const auto& hDimension = field->Dimensions.getHorizontalFieldDimension();
      if(hDimension.getType() != sir.GridType) {
        return false;
      }
    }

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