#include "TypeChecker.h"
#include <memory>

namespace dawn {

bool TypeChecker::checkLocationTypeConsistency(const dawn::iir::IIR& iir) {
  bool consistent = true;
  for(const auto& doMethodPtr : iterateIIROver<iir::DoMethod>(iir)) {
    auto const& fieldNamesToLocation = doMethodPtr->getFieldLocationsByName();
    TypeChecker::TypeCheckerImpl Impl(fieldNamesToLocation);
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

TypeChecker::TypeCheckerImpl::TypeCheckerImpl(
    const std::unordered_map<std::string, ast::Expr::LocationType>& nameToLocationMap)
    : nameToLocationType_(nameToLocationMap) {}

ast::Expr::LocationType TypeChecker::TypeCheckerImpl::getType() const {
  DAWN_ASSERT(hasType());
  return curType_.value();
}

void TypeChecker::TypeCheckerImpl::visit(
    const std::shared_ptr<iir::FieldAccessExpr>& fieldAccessExpr) {
  if(!typesConsistent_) {
    return;
  }
  DAWN_ASSERT(nameToLocationType_.count(fieldAccessExpr->getName()));
  curType_ = nameToLocationType_.at(fieldAccessExpr->getName());
}
void TypeChecker::TypeCheckerImpl::visit(const std::shared_ptr<iir::BinaryOperator>& binOp) {
  if(!typesConsistent_) {
    return;
  }

  TypeChecker::TypeCheckerImpl left(nameToLocationType_);
  TypeChecker::TypeCheckerImpl right(nameToLocationType_);

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
  TypeChecker::TypeCheckerImpl left(nameToLocationType_);
  TypeChecker::TypeCheckerImpl right(nameToLocationType_);

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

  TypeChecker::TypeCheckerImpl init(nameToLocationType_);
  TypeChecker::TypeCheckerImpl ops(nameToLocationType_);

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