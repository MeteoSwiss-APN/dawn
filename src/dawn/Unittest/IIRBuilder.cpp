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
Array3i as_array(field_type ft) {
  switch(ft) {
  case field_type::ijk:
    return Array3i{1, 1, 1};
  case field_type::ij:
    return Array3i{1, 1, 0};
  case field_type::ik:
    return Array3i{1, 0, 1};
  case field_type::jk:
    return Array3i{0, 1, 1};
  case field_type::i:
    return Array3i{1, 0, 0};
  case field_type::j:
    return Array3i{0, 1, 0};
  case field_type::k:
    return Array3i{0, 0, 1};
  }
  dawn_unreachable("Unreachable");
}
std::string to_str(op operation, std::vector<op> valid_ops) {
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
  case op::not_equal:
    return "!=";
  case op::greater:
    return ">";
  case op::less:
    return "<";
  case op::greater_equal:
    return ">=";
  case op::less_equal:
    return "<=";
  case op::logical_and:
    return "&&";
  case op::logical_or:
    return "||";
  case op::logical_not:
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
  auto stencilCallDeclStmt = std::make_shared<iir::StencilCallDeclStmt>(placeholderStencil);
  // Register the call and set it as a replacement for the next vertical region
  si_->getMetaData().addStencilCallStmt(stencilCallDeclStmt, stencil_id);

  auto stencilCallStatement = std::make_shared<Statement>(stencilCallDeclStmt, nullptr);
  si_->getIIR()->getControlFlowDescriptor().insertStmt(stencilCallStatement);

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
  auto optimizer = dawn::make_unique<dawn::OptimizerContext>(
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
      to_str(operation, {op::multiply, op::plus, op::minus, op::assign, op::divide}),
      std::move(rhs), std::move(init));
  expr->setID(si_->nextUID());
  return expr;
}
std::shared_ptr<iir::Expr> IIRBuilder::binaryExpr(std::shared_ptr<iir::Expr>&& lhs,
                                                  std::shared_ptr<iir::Expr>&& rhs, op operation) {
  DAWN_ASSERT(si_);
  auto binop = std::make_shared<iir::BinaryOperator>(
      std::move(lhs),
      to_str(operation,
             {op::multiply, op::plus, op::minus, op::divide, op::equal, op::not_equal, op::greater,
              op::less, op::greater_equal, op::less_equal, op::logical_and, op::logical_or}),
      std::move(rhs));
  binop->setID(si_->nextUID());
  return binop;
}
std::shared_ptr<iir::Expr> IIRBuilder::unaryExpr(std::shared_ptr<iir::Expr>&& expr, op operation) {
  DAWN_ASSERT(si_);
  auto ret = std::make_shared<iir::UnaryOperator>(
      std::move(expr), to_str(operation, {op::plus, op::minus, op::logical_not}));
  ret->setID(si_->nextUID());
  return ret;
}
std::shared_ptr<iir::Expr> IIRBuilder::conditionalExpr(std::shared_ptr<iir::Expr>&& cond,
                                                       std::shared_ptr<iir::Expr>&& case_then,
                                                       std::shared_ptr<iir::Expr>&& case_else) {
  DAWN_ASSERT(si_);
  auto ret = std::make_shared<iir::TernaryOperator>(std::move(cond), std::move(case_then),
                                                    std::move(case_else));
  ret->setID(si_->nextUID());
  return ret;
}
std::shared_ptr<iir::Expr> IIRBuilder::assignExpr(std::shared_ptr<iir::Expr>&& lhs,
                                                  std::shared_ptr<iir::Expr>&& rhs, op operation) {
  DAWN_ASSERT(si_);
  auto binop = std::make_shared<iir::AssignmentExpr>(
      std::move(lhs), std::move(rhs),
      to_str(operation, {op::assign, op::multiply, op::plus, op::minus, op::divide}) + "=");
  binop->setID(si_->nextUID());
  return binop;
}
IIRBuilder::Field IIRBuilder::field(std::string const& name, field_type ft) {
  DAWN_ASSERT(si_);
  int id = si_->getMetaData().addField(iir::FieldAccessType::FAT_APIField, name, as_array(ft));
  return {id, name};
}
IIRBuilder::LocalVar IIRBuilder::localvar(std::string const& name) {
  DAWN_ASSERT(si_);
  auto iir_stmt = std::make_shared<iir::VarDeclStmt>(Type{BuiltinTypeID::Float}, name, 0, "=",
                                                     std::vector<std::shared_ptr<Expr>>{});
  int id = si_->getMetaData().addStmt(true, iir_stmt);
  return {id, name, iir_stmt};
}
std::shared_ptr<iir::Expr> IIRBuilder::at(IIRBuilder::Field const& field, access_type access,
                                          Array3i extent) {
  DAWN_ASSERT(si_);
  auto expr = std::make_shared<iir::FieldAccessExpr>(field.name, extent);
  expr->setID(si_->nextUID());

  si_->getMetaData().insertExprToAccessID(expr, field.id);
  return expr;
}
std::shared_ptr<iir::Expr> IIRBuilder::at(IIRBuilder::Field const& field, Array3i extent) {
  DAWN_ASSERT(si_);
  return at(field, access_type::r, extent);
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
  auto stmt = std::make_shared<iir::ExprStmt>(std::move(expr));
  auto sap = make_unique<iir::StatementAccessesPair>(std::make_shared<Statement>(stmt, nullptr));
  return {std::move(stmt), std::move(sap)};
}
IIRBuilder::StmtData IIRBuilder::ifStmt(std::shared_ptr<iir::Expr>&& cond, StmtData&& case_then,
                                        StmtData&& case_else) {
  DAWN_ASSERT(si_);
  auto cond_stmt = std::make_shared<iir::ExprStmt>(std::move(cond));
  auto stmt = std::make_shared<iir::IfStmt>(cond_stmt, std::move(case_then.stmt),
                                            std::move(case_else.stmt));
  auto sap = make_unique<iir::StatementAccessesPair>(std::make_shared<Statement>(stmt, nullptr));
  if(case_then.sap)
    sap->insertBlockStatement(std::move(case_then.sap));
  if(case_else.sap)
    sap->insertBlockStatement(std::move(case_else.sap));
  return {std::move(stmt), std::move(sap)};
}
IIRBuilder::StmtData IIRBuilder::declareVar(IIRBuilder::LocalVar& var) {
  DAWN_ASSERT(si_);
  DAWN_ASSERT(var.decl);
  auto sap =
      make_unique<iir::StatementAccessesPair>(std::make_shared<Statement>(var.decl, nullptr));
  return {std::move(var.decl), std::move(sap)};
}

} // namespace iir
} // namespace dawn
