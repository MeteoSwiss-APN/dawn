#include "TypeChecker.h"

namespace dawn {

bool TypeChecker::checkLocationTypeConsistency(const dawn::iir::IIR& iir,
                                               const iir::StencilMetaInformation& metaData) {
  bool consistent = true;

  for(const auto& doMethodPtr : iterateIIROver<iir::DoMethod>(iir)) {
    TypeChecker::TypeCheckerImpl Impl(doMethodPtr->getFieldLocationsByName(),
                                      metaData.getAccessIDToNameMap());
    const std::shared_ptr<iir::BlockStmt>& ast =
        std::make_shared<iir::BlockStmt>(doMethodPtr->getAST());
    ast->accept(Impl);
    consistent &= Impl.isConsistent();
    if(!consistent) {
      break;
    }
  }
  return consistent;
}

bool TypeChecker::checkLocationTypeConsistency(const dawn::SIR& SIR) {
  // check type consistency of stencil functions
  bool consistent = true;
  for(auto const& stenFunIt : SIR.StencilFunctions) {
    std::unordered_map<std::string, ast::Expr::LocationType> argumentFieldLocs;
    for(const auto& arg : stenFunIt->Args) {
      if(arg->Kind == sir::StencilFunctionArg::ArgumentKind::Field) {
        const auto* argField = static_cast<sir::Field*>(arg.get());
        argumentFieldLocs.insert({argField->Name, argField->locationType});
      }
    }
    for(const auto& astIt : stenFunIt->Asts) {
      TypeChecker::TypeCheckerImpl Impl(argumentFieldLocs);
      astIt->accept(Impl);
      consistent &= Impl.isConsistent();
      if(!consistent) {
        break;
      }
    }
    if(!consistent) {
      break;
    }
  }

  // do not continue to stencils if functions aren't type consistent
  if(!consistent) {
    return false;
  }

  // check type consistency of stencils
  for(const auto& stencil : SIR.Stencils) {
    DAWN_ASSERT(stencil);
    std::unordered_map<std::string, ast::Expr::LocationType> stencilFieldLocs;
    for(const auto& field : stencil->Fields) {
      stencilFieldLocs.insert({field->Name, field->locationType});
    }
    const auto& stencilAst = stencil->StencilDescAst;
    TypeChecker::TypeCheckerImpl Impl(stencilFieldLocs);
    stencilAst->accept(Impl);
    consistent &= Impl.isConsistent();
    if(!consistent) {
      break;
    }
  }

  return consistent;
}

TypeChecker::TypeCheckerImpl::TypeCheckerImpl(
    const std::unordered_map<std::string, ast::Expr::LocationType> nameToLocationMap)
    : nameToLocationType_(nameToLocationMap) {}

TypeChecker::TypeCheckerImpl::TypeCheckerImpl(
    const std::unordered_map<std::string, ast::Expr::LocationType> nameToLocationMap,
    const std::unordered_map<int, std::string> idToNameMap)
    : nameToLocationType_(nameToLocationMap), idToNameMap_(idToNameMap) {}

ast::Expr::LocationType TypeChecker::TypeCheckerImpl::getType() const {
  DAWN_ASSERT(hasType());
  return curType_.value();
}

void TypeChecker::TypeCheckerImpl::visit(
    const std::shared_ptr<iir::FieldAccessExpr>& fieldAccessExpr) {
  if(!typesConsistent_) {
    return;
  }

  auto fieldName = fieldAccessExpr->getName();
  // the name in the FieldAccessExpr may be stale if the there are nested stencils
  // in this case we need to look up the new AccessID in the data of the fieldAccessExpr
  if(fieldAccessExpr->hasData()) {
    auto newAccessID = fieldAccessExpr->getData<iir::IIRAccessExprData>().AccessID;
    if(newAccessID.has_value()) {
      DAWN_ASSERT(idToNameMap_.count(newAccessID.value()));
      fieldName = idToNameMap_.at(newAccessID.value());
    }
  }

  DAWN_ASSERT(nameToLocationType_.count(fieldName));
  curType_ = nameToLocationType_.at(fieldName);
} // namespace dawn
void TypeChecker::TypeCheckerImpl::visit(const std::shared_ptr<iir::BinaryOperator>& binOp) {
  if(!typesConsistent_) {
    return;
  }

  TypeChecker::TypeCheckerImpl left(nameToLocationType_, idToNameMap_);
  TypeChecker::TypeCheckerImpl right(nameToLocationType_, idToNameMap_);

  binOp->getLeft()->accept(left);
  binOp->getRight()->accept(right);

  // type check failed further down below
  if(!(left.isConsistent() && right.isConsistent())) {
    typesConsistent_ = false;
  }

  // if both sides access unstructured fields, they need to be on the same location
  // one or both sides can be without type, e.g. 3*5 or 3*cell_field, so no error in this case
  if(left.hasType() && right.hasType()) {
    typesConsistent_ = left.getType() == right.getType();
    curType_ = left.getType();
  } else if(left.hasType() && !right.hasType()) {
    curType_ = left.getType();
  } else if(!left.hasType() && right.hasType()) {
    curType_ = right.getType();
  }
}
void TypeChecker::TypeCheckerImpl::visit(
    const std::shared_ptr<iir::AssignmentExpr>& assignemtExpr) {
  if(!typesConsistent_) {
    return;
  }
  TypeChecker::TypeCheckerImpl left(nameToLocationType_, idToNameMap_);
  TypeChecker::TypeCheckerImpl right(nameToLocationType_, idToNameMap_);

  assignemtExpr->getLeft()->accept(left);
  assignemtExpr->getRight()->accept(right);

  // type check failed further down below
  if(!(left.isConsistent() && right.isConsistent())) {
    typesConsistent_ = false;
  }

  // if both sides access unstructured fields, they need to be on the same location
  // one or both sides can be without type, e.g. 3*5 or 3*cell_field, so no error in this case
  if(left.hasType() && right.hasType()) {
    typesConsistent_ = left.getType() == right.getType();
    curType_ = left.getType();
  } else if(left.hasType() && !right.hasType()) {
    curType_ = left.getType();
  } else if(!left.hasType() && right.hasType()) {
    curType_ = right.getType();
  }
}
void TypeChecker::TypeCheckerImpl::visit(
    const std::shared_ptr<iir::ReductionOverNeighborExpr>& reductionExpr) {
  if(!typesConsistent_) {
    return;
  }

  TypeChecker::TypeCheckerImpl init(nameToLocationType_, idToNameMap_);
  TypeChecker::TypeCheckerImpl ops(nameToLocationType_, idToNameMap_);

  reductionExpr->getInit()->accept(init);
  reductionExpr->getRhs()->accept(ops);

  // initial value needs to be consistent with operations on right hand side
  if(init.hasType() && ops.hasType()) {
    typesConsistent_ = init.getType() == ops.getType();
  }

  // the reduce over neighbor concept imposes a type on the left hand side;
  curType_ = reductionExpr->getLhsLocation();
}

} // namespace dawn