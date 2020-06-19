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
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Validator/GridTypeChecker.h"
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
  auto optimizer = std::make_unique<dawn::OptimizerContext>(
      dawn::OptimizerContext::OptimizerContextOptions{}, nullptr);
  optimizer->restoreIIR("<restored>", std::move(si_));
  auto new_si = optimizer->getStencilInstantiationMap()["<restored>"];

  if(new_si->getIIR()->getGridType() == ast::GridType::Unstructured) {
    auto [checkResultDimensions, errorLocDimension] =
        UnstructuredDimensionChecker::checkDimensionsConsistency(*new_si->getIIR().get(),
                                                                 new_si->getMetaData());
    DAWN_ASSERT_MSG(checkResultDimensions, "Dimensions consistency check failed.");
    auto [checkResultWeights, errorLocWeights] =
        WeightChecker::CheckWeights(*new_si->getIIR().get(), new_si->getMetaData());
    DAWN_ASSERT_MSG(checkResultWeights, "Found invalid weights");
  }
  DAWN_ASSERT(GridTypeChecker::checkGridTypeConsistency(*new_si->getIIR().get()));

  dawn::codegen::StencilInstantiationContext map;
  return new_si;
}

std::shared_ptr<iir::Expr>
IIRBuilder::reduceOverNeighborExpr(Op operation, std::shared_ptr<iir::Expr>&& rhs,
                                   std::shared_ptr<iir::Expr>&& init,
                                   const std::vector<ast::LocationType>& chain) {
  auto expr = std::make_shared<iir::ReductionOverNeighborExpr>(
      toStr(operation, {Op::multiply, Op::plus, Op::minus, Op::assign, Op::divide}), std::move(rhs),
      std::move(init), chain);
  expr->setID(si_->nextUID());
  return expr;
}

std::shared_ptr<iir::Expr>
IIRBuilder::reduceOverNeighborExpr(Op operation, std::shared_ptr<iir::Expr>&& rhs,
                                   std::shared_ptr<iir::Expr>&& init,
                                   const std::vector<ast::LocationType>& chain,
                                   const std::vector<std::shared_ptr<iir::Expr>>&& weights) {
  auto expr = std::make_shared<iir::ReductionOverNeighborExpr>(
      toStr(operation, {Op::multiply, Op::plus, Op::minus, Op::assign, Op::divide}), std::move(rhs),
      std::move(init), move(weights), chain);
  expr->setID(si_->nextUID());
  return expr;
}

std::shared_ptr<iir::Expr> IIRBuilder::binaryExpr(std::shared_ptr<iir::Expr>&& lhs,
                                                  std::shared_ptr<iir::Expr>&& rhs, Op operation) {
  DAWN_ASSERT(si_);
  auto binop = std::make_shared<iir::BinaryOperator>(
      std::move(lhs),
      toStr(operation,
            {Op::multiply, Op::plus, Op::minus, Op::divide, Op::equal, Op::notEqual, Op::greater,
             Op::less, Op::greaterEqual, Op::lessEqual, Op::logicalAnd, Op::locigalOr}),
      std::move(rhs));
  binop->setID(si_->nextUID());
  return binop;
}
std::shared_ptr<iir::Expr> IIRBuilder::unaryExpr(std::shared_ptr<iir::Expr>&& expr, Op operation) {
  DAWN_ASSERT(si_);
  auto ret = std::make_shared<iir::UnaryOperator>(
      std::move(expr),
      toStr(operation, {Op::plus, Op::minus, Op::increment, Op::decrement, Op::logicalNot}));
  ret->setID(si_->nextUID());
  return ret;
}
std::shared_ptr<iir::Expr> IIRBuilder::conditionalExpr(std::shared_ptr<iir::Expr>&& cond,
                                                       std::shared_ptr<iir::Expr>&& caseThen,
                                                       std::shared_ptr<iir::Expr>&& caseElse) {
  DAWN_ASSERT(si_);
  auto ret = std::make_shared<iir::TernaryOperator>(std::move(cond), std::move(caseThen),
                                                    std::move(caseElse));
  ret->setID(si_->nextUID());
  return ret;
}
std::shared_ptr<iir::Expr> IIRBuilder::assignExpr(std::shared_ptr<iir::Expr>&& lhs,
                                                  std::shared_ptr<iir::Expr>&& rhs, Op operation) {
  DAWN_ASSERT(si_);
  auto binop = std::make_shared<iir::AssignmentExpr>(
      std::move(lhs), std::move(rhs),
      toStr(operation, {Op::assign, Op::multiply, Op::plus, Op::minus, Op::divide}) + "=");
  binop->setID(si_->nextUID());
  return binop;
}
IIRBuilder::LocalVar IIRBuilder::localvar(std::string const& name, BuiltinTypeID dataType,
                                          std::vector<std::shared_ptr<iir::Expr>>&& initList,
                                          std::optional<LocalVariableType> localVarType) {
  DAWN_ASSERT(si_);
  auto iirStmt = makeVarDeclStmt(Type{dataType}, name, 0, "=", std::move(initList));
  int id = si_->getMetaData().addStmt(true, iirStmt);
  if(localVarType.has_value()) {
    si_->getMetaData().getLocalVariableDataFromAccessID(id).setType(*localVarType);
  }
  return {id, name, iirStmt};
}
std::shared_ptr<iir::Expr> IIRBuilder::at(Field const& field, AccessType access,
                                          ast::Offsets const& offset) {
  DAWN_ASSERT(si_);
  auto expr = std::make_shared<iir::FieldAccessExpr>(field.name, offset);
  expr->setID(si_->nextUID());

  expr->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(field.id);
  return expr;
}
std::shared_ptr<iir::Expr> IIRBuilder::at(IIRBuilder::GlobalVar const& var) {
  DAWN_ASSERT(si_);
  auto expr = std::make_shared<iir::VarAccessExpr>(var.name);
  expr->setIsExternal(true);
  expr->setID(si_->nextUID());
  expr->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(var.id);
  return expr;
}
std::shared_ptr<iir::Expr> IIRBuilder::at(IIRBuilder::LocalVar const& var) {
  DAWN_ASSERT(si_);
  auto expr = std::make_shared<iir::VarAccessExpr>(var.name);
  expr->setID(si_->nextUID());
  expr->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(var.id);
  return expr;
}
std::shared_ptr<iir::Stmt> IIRBuilder::stmt(std::shared_ptr<iir::Expr>&& expr) {
  auto stmt = iir::makeExprStmt(std::move(expr));
  return stmt;
}
std::shared_ptr<iir::Stmt> IIRBuilder::ifStmt(std::shared_ptr<iir::Expr>&& cond,
                                              std::shared_ptr<iir::Stmt>&& caseThen,
                                              std::shared_ptr<iir::Stmt>&& caseElse) {
  DAWN_ASSERT(si_);
  auto condStmt = iir::makeExprStmt(std::move(cond));
  auto stmt = iir::makeIfStmt(condStmt, std::move(caseThen), std::move(caseElse));
  return stmt;
}
std::shared_ptr<iir::Stmt> IIRBuilder::loopStmtChain(std::shared_ptr<iir::BlockStmt>&& body,
                                                     std::vector<ast::LocationType>&& chain) {
  DAWN_ASSERT(si_);
  auto stmt = iir::makeLoopStmt(std::move(chain), std::move(body));
  return stmt;
}

std::shared_ptr<iir::Stmt> IIRBuilder::loopStmtChain(std::shared_ptr<iir::Stmt>&& body,
                                                     std::vector<ast::LocationType>&& chain) {
  DAWN_ASSERT(si_);
  auto bStmt = iir::makeBlockStmt(std::vector<std::shared_ptr<iir::Stmt>>{body});
  auto stmt = iir::makeLoopStmt(std::move(chain), std::move(bStmt));
  return stmt;
}

std::shared_ptr<iir::Stmt> IIRBuilder::declareVar(IIRBuilder::LocalVar& var) {
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

std::shared_ptr<iir::Expr> CartesianIIRBuilder::at(Field const& field, AccessType access) {
  return at(field, access, ast::Offsets{ast::cartesian});
}
std::shared_ptr<iir::Expr> CartesianIIRBuilder::at(IIRBuilder::Field const& field,
                                                   Array3i const& offset) {
  return at(field, AccessType::r, offset);
}
std::shared_ptr<iir::Expr> CartesianIIRBuilder::at(IIRBuilder::Field const& field,
                                                   AccessType access, Array3i const& offset) {
  return at(field, AccessType::r, ast::Offsets{ast::cartesian, offset});
}

IIRBuilder::Field UnstructuredIIRBuilder::field(std::string const& name,
                                                ast::LocationType location) {
  DAWN_ASSERT(si_);
  int id = si_->getMetaData().addField(
      iir::FieldAccessType::APIField, name,
      sir::FieldDimensions(sir::HorizontalFieldDimension{ast::unstructured, location}, true));
  return {id, name};
}

IIRBuilder::Field UnstructuredIIRBuilder::field(std::string const& name,
                                                ast::NeighborChain sparseChain) {
  DAWN_ASSERT(si_);
  int id = si_->getMetaData().addField(
      iir::FieldAccessType::APIField, name,
      sir::FieldDimensions(sir::HorizontalFieldDimension{ast::unstructured, sparseChain}, true));
  return {id, name};
}

std::shared_ptr<iir::Expr> UnstructuredIIRBuilder::at(Field const& field, AccessType access) {
  return at(field, access, ast::Offsets{ast::unstructured});
}

std::shared_ptr<iir::Expr> UnstructuredIIRBuilder::at(IIRBuilder::Field const& field,
                                                      HOffsetType hOffset, int vOffset) {
  DAWN_ASSERT(si_);
  return at(field, AccessType::r, hOffset, vOffset);
}
std::shared_ptr<iir::Expr> UnstructuredIIRBuilder::at(IIRBuilder::Field const& field,
                                                      AccessType access, HOffsetType hOffset,
                                                      int vOffset) {
  return at(field, AccessType::r,
            ast::Offsets{ast::unstructured, hOffset == HOffsetType::withOffset, vOffset});
}
} // namespace iir
} // namespace dawn
