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
//
// TODO there are death tests which rely on the following code to die, needs refactoring
#include "dawn/AST/LocationType.h"
#ifdef NDEBUG
#undef NDEBUG
#define HAD_NDEBUG
#endif
#include "dawn/Support/Assert.h"
#ifdef HAD_NDEBUG
#define NDEBUG
#undef HAD_NDEBUG
#endif

#include "IIRBuilder.h"

#include "dawn/IIR/AccessComputation.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/Optimizer/Lowering.h"
#include "dawn/Validator/GridTypeChecker.h"
#include "dawn/Validator/IntegrityChecker.h"
#include "dawn/Validator/UnstructuredDimensionChecker.h"
#include "dawn/Validator/WeightChecker.h"

namespace dawn {
namespace iir {
namespace {
Array3i asArray(FieldType ft) {
  switch(ft) {
  case FieldType::ijk:
    return Array3i{1, 1, 1};
  case FieldType::ij:
    return Array3i{1, 1, 0};
  case FieldType::ik:
    return Array3i{1, 0, 1};
  case FieldType::jk:
    return Array3i{0, 1, 1};
  case FieldType::i:
    return Array3i{1, 0, 0};
  case FieldType::j:
    return Array3i{0, 1, 0};
  case FieldType::k:
    return Array3i{0, 0, 1};
  }
  dawn_unreachable("Unreachable");
}
} // namespace

std::shared_ptr<iir::StencilInstantiation>
IIRBuilder::build(std::string const& name, std::unique_ptr<iir::Stencil> stencilIIR) {
  DAWN_ASSERT(si_);
  // setup the whole stencil instantiation
  auto stencil_id = stencilIIR->getStencilID();
  si_->getMetaData().setStencilName(name);
  si_->getIIR()->insertChild(std::move(stencilIIR), si_->getIIR());

  auto placeholderStencil = std::make_shared<ast::StencilCall>(
      iir::InstantiationHelper::makeStencilCallCodeGenName(stencil_id));
  auto stencilCallDeclStmt = iir::makeStencilCallDeclStmt(placeholderStencil);
  // Register the call and set it as a replacement for the next vertical region
  si_->getMetaData().addStencilCallStmt(stencilCallDeclStmt, stencil_id);

  si_->getIIR()->getControlFlowDescriptor().insertStmt(stencilCallDeclStmt);

  // update everything
  for(const auto& MS : iterateIIROver<iir::MultiStage>(*(si_->getIIR()))) {
    MS->update(iir::NodeUpdateType::levelAndTreeAbove);
  }
  // Iterate all statements (top -> bottom)
  for(const auto& stagePtr : iterateIIROver<iir::Stage>(*(si_->getIIR()))) {
    iir::Stage& stageIIR = *stagePtr;
    for(const auto& doMethodIIR : stageIIR.getChildren()) {
      doMethodIIR->update(iir::NodeUpdateType::level);
    }
    stageIIR.update(iir::NodeUpdateType::level);
  }
  for(const auto& stagePtr : iterateIIROver<iir::Stage>(*(si_->getIIR()))) {
    stagePtr->update(iir::NodeUpdateType::levelAndTreeAbove);
  }

  // create stencil instantiation context
  restoreIIR(si_);

  if(si_->getIIR()->getGridType() == ast::GridType::Unstructured) {
    IntegrityChecker integrityChecker(si_.get());
    integrityChecker.run();
    auto [checkResultDimensions, errorLocDimension] =
        UnstructuredDimensionChecker::checkDimensionsConsistency(*si_->getIIR().get(),
                                                                 si_->getMetaData());
    DAWN_ASSERT_MSG(checkResultDimensions, "Dimensions consistency check failed.");
    auto [checkResultWeights, errorLocWeights] =
        WeightChecker::CheckWeights(*si_->getIIR().get(), si_->getMetaData());
    DAWN_ASSERT_MSG(checkResultWeights, "Found invalid weights");
  }
  DAWN_ASSERT(GridTypeChecker::checkGridTypeConsistency(*si_->getIIR().get()));

  dawn::codegen::StencilInstantiationContext map;
  return si_;
}

std::shared_ptr<ast::Expr> IIRBuilder::reduceOverNeighborExpr(
    Op operation, std::shared_ptr<ast::Expr>&& rhs, std::shared_ptr<ast::Expr>&& init,
    const std::vector<ast::LocationType>& chain, bool includeCenter) {
  auto expr = std::make_shared<ast::ReductionOverNeighborExpr>(
      toStr(operation, {Op::multiply, Op::plus, Op::minus, Op::assign, Op::divide}), std::move(rhs),
      std::move(init), chain, includeCenter);
  expr->setID(si_->nextUID());
  return expr;
}

std::shared_ptr<ast::Expr> IIRBuilder::reduceOverNeighborExpr(
    Op operation, std::shared_ptr<ast::Expr>&& rhs, std::shared_ptr<ast::Expr>&& init,
    const std::vector<ast::LocationType>& chain,
    const std::vector<std::shared_ptr<ast::Expr>>&& weights, bool includeCenter) {
  auto expr = std::make_shared<ast::ReductionOverNeighborExpr>(
      toStr(operation, {Op::multiply, Op::plus, Op::minus, Op::assign, Op::divide}), std::move(rhs),
      std::move(init), move(weights), chain, includeCenter);
  expr->setID(si_->nextUID());
  return expr;
}

std::shared_ptr<ast::Expr> IIRBuilder::binaryExpr(std::shared_ptr<ast::Expr>&& lhs,
                                                  std::shared_ptr<ast::Expr>&& rhs, Op operation) {
  DAWN_ASSERT(si_);
  auto binop = std::make_shared<ast::BinaryOperator>(
      std::move(lhs),
      toStr(operation,
            {Op::multiply, Op::plus, Op::minus, Op::divide, Op::equal, Op::notEqual, Op::greater,
             Op::less, Op::greaterEqual, Op::lessEqual, Op::logicalAnd, Op::locigalOr}),
      std::move(rhs));
  binop->setID(si_->nextUID());
  return binop;
}
std::shared_ptr<ast::Expr> IIRBuilder::unaryExpr(std::shared_ptr<ast::Expr>&& expr, Op operation) {
  DAWN_ASSERT(si_);
  auto ret = std::make_shared<ast::UnaryOperator>(
      std::move(expr),
      toStr(operation, {Op::plus, Op::minus, Op::increment, Op::decrement, Op::logicalNot}));
  ret->setID(si_->nextUID());
  return ret;
}
std::shared_ptr<ast::Expr> IIRBuilder::conditionalExpr(std::shared_ptr<ast::Expr>&& cond,
                                                       std::shared_ptr<ast::Expr>&& caseThen,
                                                       std::shared_ptr<ast::Expr>&& caseElse) {
  DAWN_ASSERT(si_);
  auto ret = std::make_shared<ast::TernaryOperator>(std::move(cond), std::move(caseThen),
                                                    std::move(caseElse));
  ret->setID(si_->nextUID());
  return ret;
}
std::shared_ptr<ast::Expr> IIRBuilder::assignExpr(std::shared_ptr<ast::Expr>&& lhs,
                                                  std::shared_ptr<ast::Expr>&& rhs, Op operation) {
  DAWN_ASSERT(si_);
  auto binop = std::make_shared<ast::AssignmentExpr>(
      std::move(lhs), std::move(rhs),
      toStr(operation, {Op::assign, Op::multiply, Op::plus, Op::minus, Op::divide}) + "=");
  binop->setID(si_->nextUID());
  return binop;
}
IIRBuilder::LocalVar IIRBuilder::localvar(std::string const& name, BuiltinTypeID dataType,
                                          std::vector<std::shared_ptr<ast::Expr>>&& initList,
                                          std::optional<LocalVariableType> localVarType) {
  DAWN_ASSERT(si_);
  auto iirStmt = makeVarDeclStmt(Type{dataType}, name, 0, "=", std::move(initList));
  int id = si_->getMetaData().addStmt(true, iirStmt);
  if(localVarType.has_value()) {
    si_->getMetaData().getLocalVariableDataFromAccessID(id).setType(*localVarType);
  }
  return {id, name, iirStmt};
}
std::shared_ptr<ast::Expr> IIRBuilder::at(Field const& field, AccessType access,
                                          ast::Offsets const& offset) {
  DAWN_ASSERT(si_);
  auto expr = std::make_shared<ast::FieldAccessExpr>(field.name, offset);
  expr->setID(si_->nextUID());
  if(offset.hasVerticalIndirection()) {
    expr->getOffset().setVerticalIndirectionAccessID(
        si_->getMetaData().getNameToAccessIDMap().at(offset.getVerticalIndirectionFieldName()));
  }
  expr->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(field.id);
  return expr;
}
std::shared_ptr<ast::Expr> IIRBuilder::at(IIRBuilder::GlobalVar const& var) {
  DAWN_ASSERT(si_);
  auto expr = std::make_shared<ast::VarAccessExpr>(var.name);
  expr->setIsExternal(true);
  expr->setID(si_->nextUID());
  expr->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(var.id);
  return expr;
}
std::shared_ptr<ast::Expr> IIRBuilder::at(IIRBuilder::LocalVar const& var) {
  DAWN_ASSERT(si_);
  auto expr = std::make_shared<ast::VarAccessExpr>(var.name);
  expr->setID(si_->nextUID());
  expr->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(var.id);
  return expr;
}
std::shared_ptr<ast::Stmt> IIRBuilder::stmt(std::shared_ptr<ast::Expr>&& expr) {
  auto stmt = iir::makeExprStmt(std::move(expr));
  return stmt;
}
std::shared_ptr<ast::Stmt> IIRBuilder::ifStmt(std::shared_ptr<ast::Expr>&& cond,
                                              std::shared_ptr<ast::Stmt>&& caseThen,
                                              std::shared_ptr<ast::Stmt>&& caseElse) {
  DAWN_ASSERT(si_);
  auto condStmt = iir::makeExprStmt(std::move(cond));
  auto stmt = iir::makeIfStmt(condStmt, std::move(caseThen), std::move(caseElse));
  return stmt;
}
std::shared_ptr<ast::Stmt> IIRBuilder::loopStmtChain(std::shared_ptr<ast::BlockStmt>&& body,
                                                     std::vector<ast::LocationType>&& chain,
                                                     bool includeCenter) {
  DAWN_ASSERT(si_);
  auto stmt = iir::makeLoopStmt(std::move(chain), includeCenter, std::move(body));
  return stmt;
}

std::shared_ptr<ast::Stmt> IIRBuilder::loopStmtChain(std::shared_ptr<ast::Stmt>&& body,
                                                     std::vector<ast::LocationType>&& chain,
                                                     bool includeCenter) {
  DAWN_ASSERT(si_);
  auto bStmt = iir::makeBlockStmt(std::vector<std::shared_ptr<ast::Stmt>>{body});
  auto stmt = iir::makeLoopStmt(std::move(chain), includeCenter, std::move(bStmt));
  return stmt;
}

std::shared_ptr<ast::Stmt> IIRBuilder::declareVar(IIRBuilder::LocalVar& var) {
  DAWN_ASSERT(si_);
  DAWN_ASSERT(var.decl);
  return var.decl;
}

IIRBuilder::Field CartesianIIRBuilder::field(const std::string& name, FieldType ft) {
  DAWN_ASSERT(si_);
  auto fieldMaskArray = asArray(ft);
  sir::FieldDimensions dimensions(
      sir::HorizontalFieldDimension{ast::cartesian,
                                    {fieldMaskArray[0] == 1, fieldMaskArray[1] == 1}},
      fieldMaskArray[2] == 1);
  int id = si_->getMetaData().addField(iir::FieldAccessType::APIField, name, std::move(dimensions));
  return {id, name};
}

IIRBuilder::Field CartesianIIRBuilder::tmpField(const std::string& name, FieldType ft) {
  DAWN_ASSERT(si_);
  auto fieldMaskArray = asArray(ft);
  sir::FieldDimensions dimensions(
      sir::HorizontalFieldDimension{ast::cartesian,
                                    {fieldMaskArray[0] == 1, fieldMaskArray[1] == 1}},
      fieldMaskArray[2] == 1);
  int id = si_->getMetaData().addTmpField(iir::FieldAccessType::StencilTemporary, name,
                                          std::move(dimensions));
  std::string newName = si_->getMetaData().getFieldNameFromAccessID(id);
  return {id, newName};
}

std::shared_ptr<ast::Expr> CartesianIIRBuilder::at(Field const& field, AccessType access) {
  return at(field, access, ast::Offsets{ast::cartesian});
}
std::shared_ptr<ast::Expr> CartesianIIRBuilder::at(IIRBuilder::Field const& field,
                                                   Array3i const& offset) {
  return at(field, AccessType::r, offset);
}
std::shared_ptr<ast::Expr> CartesianIIRBuilder::at(IIRBuilder::Field const& field,
                                                   AccessType access, Array3i const& offset) {
  return at(field, AccessType::r, ast::Offsets{ast::cartesian, offset});
}

IIRBuilder::Field UnstructuredIIRBuilder::field(std::string const& name, ast::LocationType location,
                                                bool maskK, bool includeCenter) {
  DAWN_ASSERT(si_);
  int id = si_->getMetaData().addField(
      iir::FieldAccessType::APIField, name,
      sir::FieldDimensions(
          sir::HorizontalFieldDimension{ast::unstructured, location, includeCenter}, maskK));
  return {id, name};
}

IIRBuilder::Field UnstructuredIIRBuilder::field(std::string const& name,
                                                ast::NeighborChain sparseChain, bool maskK,
                                                bool includeCenter) {
  DAWN_ASSERT(si_);
  int id = si_->getMetaData().addField(
      iir::FieldAccessType::APIField, name,
      sir::FieldDimensions(
          sir::HorizontalFieldDimension{ast::unstructured, sparseChain, includeCenter}, maskK));
  return {id, name};
}

IIRBuilder::Field UnstructuredIIRBuilder::tmpField(std::string const& name,
                                                   ast::LocationType location, bool maskK,
                                                   bool includeCenter) {
  DAWN_ASSERT(si_);
  int id = si_->getMetaData().addField(
      iir::FieldAccessType::StencilTemporary, name,
      sir::FieldDimensions(
          sir::HorizontalFieldDimension{ast::unstructured, location, includeCenter}, maskK));
  return {id, name};
}

IIRBuilder::Field UnstructuredIIRBuilder::tmpField(std::string const& name,
                                                   ast::NeighborChain sparseChain, bool maskK,
                                                   bool includeCenter) {
  DAWN_ASSERT(si_);
  int id = si_->getMetaData().addField(
      iir::FieldAccessType::StencilTemporary, name,
      sir::FieldDimensions(
          sir::HorizontalFieldDimension{ast::unstructured, sparseChain, includeCenter}, maskK));
  return {id, name};
}

IIRBuilder::Field UnstructuredIIRBuilder::vertical_field(std::string const& name) {
  DAWN_ASSERT(si_);
  int id =
      si_->getMetaData().addField(iir::FieldAccessType::APIField, name, sir::FieldDimensions(true));
  return {id, name};
}

std::shared_ptr<ast::Expr> UnstructuredIIRBuilder::at(Field const& field, AccessType access) {
  return at(field, access, ast::Offsets{ast::unstructured});
}

std::shared_ptr<ast::Expr> UnstructuredIIRBuilder::at(IIRBuilder::Field const& field,
                                                      HOffsetType hOffset, int vOffset) {
  DAWN_ASSERT(si_);
  return at(field, AccessType::r, hOffset, vOffset);
}
std::shared_ptr<ast::Expr> UnstructuredIIRBuilder::at(IIRBuilder::Field const& field,
                                                      AccessType access, HOffsetType hOffset,
                                                      int vOffset) {
  return at(field, AccessType::r,
            ast::Offsets{ast::unstructured, hOffset == HOffsetType::withOffset, vOffset});
}
} // namespace iir
} // namespace dawn
