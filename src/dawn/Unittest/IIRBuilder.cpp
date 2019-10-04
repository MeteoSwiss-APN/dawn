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
#include "IIRBuilder.h"

#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/Optimizer/OptimizerContext.h"

namespace dawn {
namespace iir {
namespace {
Array3i asArray(fieldType ft) {
  switch(ft) {
  case fieldType::ijk:
    return Array3i{1, 1, 1};
  case fieldType::ij:
    return Array3i{1, 1, 0};
  case fieldType::ik:
    return Array3i{1, 0, 1};
  case fieldType::jk:
    return Array3i{0, 1, 1};
  case fieldType::i:
    return Array3i{1, 0, 0};
  case fieldType::j:
    return Array3i{0, 1, 0};
  case fieldType::k:
    return Array3i{0, 0, 1};
  }
  dawn_unreachable("Unreachable");
}
std::string toStr(op operation, std::vector<op> const& valid_ops) {
  DAWN_ASSERT(std::find(valid_ops.begin(), valid_ops.end(), operation) != valid_ops.end());
  switch(operation) {
  case op::plus:
    return "+";
  case op::minus:
    return "-";
  case op::multiply:
    return "*";
  case op::assign:
    return "";
  case op::divide:
    return "/";
  case op::equal:
    return "==";
  case op::notEqual:
    return "!=";
  case op::greater:
    return ">";
  case op::less:
    return "<";
  case op::greaterEqual:
    return ">=";
  case op::lessEqual:
    return "<=";
  case op::logicalAnd:
    return "&&";
  case op::locigalOr:
    return "||";
  case op::logicalNot:
    return "!";
  }
  dawn_unreachable("Unreachable");
}
} // namespace

dawn::codegen::stencilInstantiationContext
IIRBuilder::build(std::string const& name, std::unique_ptr<iir::Stencil> stencil) {
  DAWN_ASSERT(si_);
  // setup the whole stencil instantiation
  auto stencil_id = stencil->getStencilID();
  si_->getMetaData().setStencilname(name);
  si_->getIIR()->insertChild(std::move(stencil), si_->getIIR());

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
    iir::Stage& stage = *stagePtr;
    for(const auto& doMethod : stage.getChildren()) {
      doMethod->update(iir::NodeUpdateType::level);
    }
    stage.update(iir::NodeUpdateType::level);
  }
  for(const auto& MSPtr : iterateIIROver<iir::Stage>(*(si_->getIIR()))) {
    MSPtr->update(iir::NodeUpdateType::levelAndTreeAbove);
  }

  // create stencil instantiation context
  dawn::DiagnosticsEngine diagnostics;
  auto optimizer = std::make_unique<dawn::OptimizerContext>(
      diagnostics, dawn::OptimizerContext::OptimizerContextOptions{}, nullptr);
  optimizer->restoreIIR("<restored>", std::move(si_));
  auto new_si = optimizer->getStencilInstantiationMap()["<restored>"];

  dawn::codegen::stencilInstantiationContext map;
  map[new_si->getName()] = std::move(new_si);

  return map;
}
std::shared_ptr<iir::Expr> IIRBuilder::reduceOverNeighborExpr(op operation,
                                                              std::shared_ptr<iir::Expr>&& rhs,
                                                              std::shared_ptr<iir::Expr>&& init) {
  auto expr = std::make_shared<iir::ReductionOverNeighborExpr>(
      toStr(operation, {op::multiply, op::plus, op::minus, op::assign, op::divide}), std::move(rhs),
      std::move(init));
  expr->setID(si_->nextUID());
  return expr;
}
std::shared_ptr<iir::Expr> IIRBuilder::binaryExpr(std::shared_ptr<iir::Expr>&& lhs,
                                                  std::shared_ptr<iir::Expr>&& rhs, op operation) {
  DAWN_ASSERT(si_);
  auto binop = std::make_shared<iir::BinaryOperator>(
      std::move(lhs),
      toStr(operation,
            {op::multiply, op::plus, op::minus, op::divide, op::equal, op::notEqual, op::greater,
             op::less, op::greaterEqual, op::lessEqual, op::logicalAnd, op::locigalOr}),
      std::move(rhs));
  binop->setID(si_->nextUID());
  return binop;
}
std::shared_ptr<iir::Expr> IIRBuilder::unaryExpr(std::shared_ptr<iir::Expr>&& expr, op operation) {
  DAWN_ASSERT(si_);
  auto ret = std::make_shared<iir::UnaryOperator>(
      std::move(expr), toStr(operation, {op::plus, op::minus, op::logicalNot}));
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
                                                  std::shared_ptr<iir::Expr>&& rhs, op operation) {
  DAWN_ASSERT(si_);
  auto binop = std::make_shared<iir::AssignmentExpr>(
      std::move(lhs), std::move(rhs),
      toStr(operation, {op::assign, op::multiply, op::plus, op::minus, op::divide}) + "=");
  binop->setID(si_->nextUID());
  return binop;
}
IIRBuilder::Field IIRBuilder::field(std::string const& name, fieldType ft) {
  DAWN_ASSERT(si_);
  int id = si_->getMetaData().addField(iir::FieldAccessType::FAT_APIField, name, asArray(ft));
  return {id, name, false, dawn::ast::FieldAccessExpr::Location::CELLS};
}
IIRBuilder::Field IIRBuilder::field(std::string const& name, dawn::ast::FieldAccessExpr::Location location) {
  DAWN_ASSERT(si_);
  int id = si_->getMetaData().addField(iir::FieldAccessType::FAT_APIField, name, asArray(fieldType::ijk));
  si_->getMetaData().addUnorderedField(id, location);
  return {id, name, true, location};
}
IIRBuilder::LocalVar IIRBuilder::localvar(std::string const& name) {
  DAWN_ASSERT(si_);
  auto iirStmt = makeVarDeclStmt(Type{BuiltinTypeID::Float}, name, 0, "=",
                                 std::vector<std::shared_ptr<Expr>>{});
  int id = si_->getMetaData().addStmt(true, iirStmt);
  return {id, name, iirStmt};
}
std::shared_ptr<iir::Expr> IIRBuilder::at(IIRBuilder::Field const& field, accessType access,
                                          Array3i extent) {
  DAWN_ASSERT(si_);
  auto expr = std::make_shared<iir::FieldAccessExpr>(field.name, extent);
  expr->setID(si_->nextUID());

  si_->getMetaData().insertExprToAccessID(expr, field.id);
  return expr;
}
std::shared_ptr<iir::Expr> IIRBuilder::at(IIRBuilder::Field const& field, Array3i extent) {
  DAWN_ASSERT(si_);
  return at(field, accessType::r, extent);
}
std::shared_ptr<iir::Expr> IIRBuilder::at(IIRBuilder::LocalVar const& var) {
  DAWN_ASSERT(si_);
  auto expr = std::make_shared<iir::VarAccessExpr>(var.name);
  expr->setID(si_->nextUID());
  si_->getMetaData().insertExprToAccessID(expr, var.id);
  return expr;
}
IIRBuilder::StmtData IIRBuilder::stmt(std::shared_ptr<iir::Expr>&& expr) {
  DAWN_ASSERT(si_);
  auto stmt = iir::makeExprStmt(std::move(expr));
  auto sap = std::make_unique<iir::StatementAccessesPair>(stmt);
  return {std::move(stmt), std::move(sap)};
}
IIRBuilder::StmtData IIRBuilder::ifStmt(std::shared_ptr<iir::Expr>&& cond, StmtData&& caseThen,
                                        StmtData&& caseElse) {
  DAWN_ASSERT(si_);
  auto condStmt = iir::makeExprStmt(std::move(cond));
  auto stmt = iir::makeIfStmt(condStmt, std::move(caseThen.stmt), std::move(caseElse.stmt));
  auto sap = std::make_unique<iir::StatementAccessesPair>(stmt);
  if(caseThen.sap)
    sap->insertBlockStatement(std::move(caseThen.sap));
  if(caseElse.sap)
    sap->insertBlockStatement(std::move(caseElse.sap));
  return {std::move(stmt), std::move(sap)};
}
IIRBuilder::StmtData IIRBuilder::declareVar(IIRBuilder::LocalVar& var) {
  DAWN_ASSERT(si_);
  DAWN_ASSERT(var.decl);
  auto sap = std::make_unique<iir::StatementAccessesPair>(var.decl);
  return {std::move(var.decl), std::move(sap)};
}

} // namespace iir
} // namespace dawn
