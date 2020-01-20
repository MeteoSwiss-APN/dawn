#include "TypeChecker.h"

namespace dawn {

bool TypeChecker::checkDimensionsConsistency(const dawn::iir::IIR& iir,
                                             const iir::StencilMetaInformation& metaData) {
  for(const auto& doMethodPtr : iterateIIROver<iir::DoMethod>(iir)) {
    TypeChecker::TypeCheckerImpl Impl(doMethodPtr->getFieldDimensionsByName(),
                                      metaData.getAccessIDToNameMap());
    const std::shared_ptr<iir::BlockStmt>& ast =
        std::make_shared<iir::BlockStmt>(doMethodPtr->getAST());
    ast->accept(Impl);
    if(!Impl.isConsistent()) {
      return false;
    }
  }
  return true;
}

bool TypeChecker::checkDimensionsConsistency(const dawn::SIR& SIR) {
  // check type consistency of stencil functions
  for(auto const& stenFunIt : SIR.StencilFunctions) {
    std::unordered_map<std::string, sir::FieldDimensions> argumentFieldDimensions;
    for(const auto& arg : stenFunIt->Args) {
      if(arg->Kind == sir::StencilFunctionArg::ArgumentKind::Field) {
        const auto* argField = static_cast<sir::Field*>(arg.get());
        argumentFieldDimensions.insert({argField->Name, argField->Dimensions});
      }
    }
    for(const auto& astIt : stenFunIt->Asts) {
      TypeChecker::TypeCheckerImpl Impl(argumentFieldDimensions);
      astIt->accept(Impl);
      if(!Impl.isConsistent()) {
        return false;
      }
    }
  }

  // check type consistency of stencils
  for(const auto& stencil : SIR.Stencils) {
    DAWN_ASSERT(stencil);
    std::unordered_map<std::string, sir::FieldDimensions> stencilFieldDims;
    for(const auto& field : stencil->Fields) {
      stencilFieldDims.insert({field->Name, field->Dimensions});
    }
    const auto& stencilAst = stencil->StencilDescAst;
    TypeChecker::TypeCheckerImpl Impl(stencilFieldDims);
    stencilAst->accept(Impl);
    if(!Impl.isConsistent()) {
      return false;
    }
  }

  return true;
}

///@brief Helper functions
namespace {

bool isHorizontalTriangular(const sir::FieldDimensions& dims) {
  return sir::dimension_isa<sir::TriangularFieldDimension>(dims.getHorizontalFieldDimension());
}

const sir::TriangularFieldDimension& getTriangularDim(const sir::FieldDimensions& dims) {
  return sir::dimension_cast<const sir::TriangularFieldDimension&>(
      dims.getHorizontalFieldDimension());
}

} // namespace

TypeChecker::TypeCheckerImpl::TypeCheckerImpl(
    const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap)
    : nameToDimensions_(nameToDimensionsMap) {}

TypeChecker::TypeCheckerImpl::TypeCheckerImpl(
    const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap,
    const std::unordered_map<int, std::string> idToNameMap)
    : nameToDimensions_(nameToDimensionsMap), idToNameMap_(idToNameMap) {}

const sir::FieldDimensions& TypeChecker::TypeCheckerImpl::getDimensions() const {
  DAWN_ASSERT(hasDimensions());
  return curDimensions_.value();
}

void TypeChecker::TypeCheckerImpl::visit(
    const std::shared_ptr<iir::FieldAccessExpr>& fieldAccessExpr) {
  if(!dimensionsConsistent_) {
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

  DAWN_ASSERT(nameToDimensions_.count(fieldName));
  curDimensions_ = nameToDimensions_.at(fieldName);
}

void TypeChecker::TypeCheckerImpl::checkBinaryOpTriangular(const sir::FieldDimensions& left,
                                                           const sir::FieldDimensions& right) {
  const auto& triangularDimLeft = getTriangularDim(left);
  const auto& triangularDimRight = getTriangularDim(right);

  // Case 0: Both operands are sparse. Their dense + sparse parts must match.
  // example:
  // ```
  // field(edges, e->c->v) sparseL, sparseR;
  // ...
  // expr = (sparseL + sparseR);
  // ```
  // Result of expression (= iteration space) is sparse dimension e->c->v
  if(triangularDimLeft.isSparse() && triangularDimRight.isSparse()) {
    // Check that neighbor chains match
    if(triangularDimLeft.getNeighborChain() != triangularDimRight.getNeighborChain()) {
      dimensionsConsistent_ = false;
      return;
    }
    curDimensions_ = left; // pick one, they are the same
  } else if(triangularDimLeft.isDense() && triangularDimRight.isSparse()) {

    // Case 1.a: Left is dense, right is sparse, left's location type matches right's target
    // location type (last of chain).
    // Example:
    // ```
    // field(vertices) denseL;
    // field(edges, e->c->v) sparseR;
    // ...
    // expr = (denseL + sparseR);
    // ```
    // Result of expression (= iteration space) is sparse dimension: e->c->v
    if(triangularDimLeft.getDenseLocationType() == triangularDimRight.getLastSparseLocationType()) {
      // Propagate sparse
      curDimensions_ = right;
    }
    // Case 1.b: Left is dense, right is sparse, left's location type matches right's dense
    // location type (first of chain).
    // Example:
    // ```
    // field(edges) denseL;
    // field(edges, e->c->v) sparseR;
    // ...
    // expr = (denseL + sparseR);
    // ```
    // Result of expression (= iteration space) is sparse dimension: e->c->v
    else if(triangularDimLeft.getDenseLocationType() == triangularDimRight.getDenseLocationType()) {
      // Propagate sparse
      curDimensions_ = right;

    } else { // No other subcase is allowed.
      dimensionsConsistent_ = false;
    }

  } else if(triangularDimLeft.isSparse() && triangularDimRight.isDense()) {

    // Case 2.a: Left is sparse, right is dense, right's location type matches left's target
    // location type (last of chain).
    // Example:
    // ```
    // field(edges, e->c->v) sparseL;
    // field(vertices) denseR;
    // ...
    // expr = (sparseL + denseR);
    // ```
    // Result of expression (= iteration space) is sparse dimension: e->c->v
    if(triangularDimLeft.getLastSparseLocationType() == triangularDimRight.getDenseLocationType()) {
      // Propagate sparse
      curDimensions_ = left;
    }
    // Case 2.b: Left is sparse, right is dense, right's location type matches left's dense
    // location type (first of chain).
    // Example:
    // ```
    // field(edges, e->c->v) sparseL;
    // field(edges) denseR;
    // ...
    // expr = (sparseL + denseR);
    // ```
    // Result of expression (= iteration space) is sparse dimension: e->c->v
    else if(triangularDimLeft.getDenseLocationType() == triangularDimRight.getDenseLocationType()) {
      // Propagate sparse
      curDimensions_ = left;

    } else { // No other subcase is allowed.
      dimensionsConsistent_ = false;
    }
  }
  // Case 3: Both operands are dense. They must match.
  // example:
  // ```
  // field(edges) denseL, denseR;
  // ...
  // expr = (denseL + denseR);
  // ```
  // Result of expression (= iteration space) is dense dimension edges
  else if(triangularDimLeft.isDense() && triangularDimRight.isDense()) {

    if(triangularDimLeft.getDenseLocationType() != triangularDimRight.getDenseLocationType()) {
      dimensionsConsistent_ = false;
      return;
    }
    curDimensions_ = left; // pick one, they are the same

  } else {
    dawn_unreachable("All cases should be covered.");
  }
}

void TypeChecker::TypeCheckerImpl::visit(const std::shared_ptr<iir::BinaryOperator>& binOp) {
  if(!dimensionsConsistent_) {
    return;
  }

  TypeChecker::TypeCheckerImpl left(nameToDimensions_, idToNameMap_);
  TypeChecker::TypeCheckerImpl right(nameToDimensions_, idToNameMap_);

  binOp->getLeft()->accept(left);
  binOp->getRight()->accept(right);

  // type check failed further down below
  if(!(left.isConsistent() && right.isConsistent())) {
    dimensionsConsistent_ = false;
    return;
  }

  // if both sides access unstructured fields, they need to be on the same target location type
  // or both sides can be without type, e.g. 3*5 or 3*cell_field, so no error in this case
  if(left.hasDimensions() && right.hasDimensions()) {
    if(isHorizontalTriangular(
           left.getDimensions())) { // Assuming previous checks on gridtype consistency (right must
                                    // also be triangular)

      checkBinaryOpTriangular(left.getDimensions(), right.getDimensions());

    } else { // Cartesian
      curDimensions_ = left.getDimensions();
    }
  } else if(left.hasDimensions() && !right.hasDimensions()) {
    curDimensions_ = left.getDimensions();
  } else if(!left.hasDimensions() && right.hasDimensions()) {
    curDimensions_ = right.getDimensions();
  }
}
void TypeChecker::TypeCheckerImpl::visit(
    const std::shared_ptr<iir::AssignmentExpr>& assignmentExpr) {
  if(!dimensionsConsistent_) {
    return;
  }
  TypeChecker::TypeCheckerImpl left(nameToDimensions_, idToNameMap_);
  TypeChecker::TypeCheckerImpl right(nameToDimensions_, idToNameMap_);

  assignmentExpr->getLeft()->accept(left);
  assignmentExpr->getRight()->accept(right);

  // type check failed further down below
  if(!(left.isConsistent() && right.isConsistent())) {
    dimensionsConsistent_ = false;
    return;
  }

  // if both sides access unstructured fields, they need to be on the same target location type
  // or both sides can be without type, e.g. 3*5 or 3*cell_field, so no error in this case
  if(left.hasDimensions() && right.hasDimensions()) {
    if(isHorizontalTriangular(
           left.getDimensions())) { // Assuming previous checks on gridtype consistency (right must
      // also be triangular)
      const auto& triangularDimLeft = getTriangularDim(left.getDimensions());
      const auto& triangularDimRight = getTriangularDim(right.getDimensions());

      // Case 0: Both lhs and rhs are sparse. Their dense + sparse parts must match.
      // example:
      // ```
      // field(edges, e->c->v) sparseL, sparseR;
      // ...
      // sparseL = sparseR;
      // ```
      // Result of assignment (= iteration space) is sparse dimension e->c->v
      if(triangularDimLeft.isSparse() && triangularDimRight.isSparse()) {
        // Check that neighbor chains match
        dimensionsConsistent_ =
            triangularDimLeft.getNeighborChain() == triangularDimRight.getNeighborChain();

      }
      // Lhs dense and rhs sparse is not possible, because there would be multiple values to write
      // on the same location.
      else if(triangularDimLeft.isDense() && triangularDimRight.isSparse()) {
        dimensionsConsistent_ = false;
      } else if(triangularDimLeft.isSparse() && triangularDimRight.isDense()) {

        // Case 1.a: Lhs is sparse, rhs is dense, rhs's location type matches lhs's target
        // location type (last of chain).
        // Example:
        // ```
        // field(edges, e->c->v) sparseL;
        // field(vertices) denseR;
        // ...
        // sparseL = denseR;
        // ```
        // Result of expression (= iteration space) is sparse dimension: e->c->v
        if(triangularDimLeft.getLastSparseLocationType() ==
           triangularDimRight.getDenseLocationType()) {
          // Nothing to do here
        }
        // Case 1.b: Lhs is sparse, rhs is dense, rhs's location type matches lhs's dense
        // location type (first of chain).
        // Example:
        // ```
        // field(edges, e->c->v) sparseL;
        // field(edges) denseR;
        // ...
        // sparseL = denseR;
        // ```
        // Result of expression (= iteration space) is sparse dimension: e->c->v
        else if(triangularDimLeft.getDenseLocationType() ==
                triangularDimRight.getDenseLocationType()) {
          // Nothing to do here

        } else { // No other subcase is allowed.
          dimensionsConsistent_ = false;
        }
      }
      // Case 2: Both operands are dense. They must match.
      // example:
      // ```
      // field(edges) denseL, denseR;
      // ...
      // denseL = denseR;
      // ```
      // Result of expression (= iteration space) is dense dimension edges
      else if(triangularDimLeft.isDense() && triangularDimRight.isDense()) {

        dimensionsConsistent_ =
            triangularDimLeft.getDenseLocationType() == triangularDimRight.getDenseLocationType();

      } else {
        dawn_unreachable("All cases should be covered.");
      }
    }

    if(!dimensionsConsistent_) {
      return;
    }

    // Dimensions to propagate are always those of the lhs (if the lhs has dimensions)
    curDimensions_ = left.getDimensions();

  } else if(left.hasDimensions() && !right.hasDimensions()) {
    curDimensions_ = left.getDimensions();
  } else if(!left.hasDimensions() && right.hasDimensions()) {
    curDimensions_ = right.getDimensions();
  }
}
void TypeChecker::TypeCheckerImpl::visit(
    const std::shared_ptr<iir::ReductionOverNeighborExpr>& reductionExpr) {
  if(!dimensionsConsistent_) {
    return;
  }

  TypeChecker::TypeCheckerImpl init(nameToDimensions_, idToNameMap_);
  TypeChecker::TypeCheckerImpl ops(nameToDimensions_, idToNameMap_);

  reductionExpr->getInit()->accept(init);
  reductionExpr->getRhs()->accept(ops);

  // initial value needs to be consistent with operations on right hand side
  if(init.hasDimensions() && ops.hasDimensions()) {
    // As init and rhs get combined through a binary operation, let's reuse the same code
    checkBinaryOpTriangular(init.getDimensions(), ops.getDimensions());
  }

  if(!dimensionsConsistent_) {
    return;
  }

  // if the rhs subtree has dimensions, we must check that such dimensions are consistent with the
  // declared rhs and lhs location types
  if(ops.hasDimensions()) {
    const auto& rhsTriangularDim = getTriangularDim(ops.getDimensions());
    dimensionsConsistent_ =
        rhsTriangularDim.isSparse()
            ? (rhsTriangularDim.getLastSparseLocationType() == reductionExpr->getRhsLocation() &&
               rhsTriangularDim.getDenseLocationType() == reductionExpr->getLhsLocation())
            : (rhsTriangularDim.getDenseLocationType() == reductionExpr->getRhsLocation());
  }

  if(!dimensionsConsistent_) {
    return;
  }

  // the reduce over neighbors concept imposes a type on the left hand side
  curDimensions_ = sir::FieldDimensions(
      sir::HorizontalFieldDimension(ast::triangular, reductionExpr->getLhsLocation()),
      ops.hasDimensions() ? ops.getDimensions().K() : false);
}

} // namespace dawn