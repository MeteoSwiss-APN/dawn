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
#include "InstantiationHelper.h"
#include "StatementAccessesPair.h"

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
  return {};
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
  default:
    DAWN_ASSERT(false);
    return "";
  }
}
} // namespace

std::shared_ptr<iir::StencilInstantiation>
IIRBuilder::build(std::string const& name, std::unique_ptr<iir::Stencil> stencil) {
  auto stencil_id = stencil->getStencilID();
  si_->getMetaData().setStencilname(name);
  si_->getIIR()->insertChild(std::move(stencil), si_->getIIR());

  auto stencilCall = std::make_shared<ast::StencilCall>("generatedDriver");
  // stencilCall->Args.push_back(sirInField->Name);
  // stencilCall->Args.push_back(sirOutField->Name);
  auto placeholderStencil = std::make_shared<ast::StencilCall>(
      iir::InstantiationHelper::makeStencilCallCodeGenName(stencil_id));
  auto stencilCallDeclStmt = std::make_shared<iir::StencilCallDeclStmt>(placeholderStencil);
  // Register the call and set it as a replacement for the next vertical region
  si_->getMetaData().addStencilCallStmt(stencilCallDeclStmt, stencil_id);

  auto stencilCallStatement = std::make_shared<Statement>(stencilCallDeclStmt, nullptr);
  si_->getIIR()->getControlFlowDescriptor().insertStmt(stencilCallStatement);

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
  return std::move(si_);
}
std::shared_ptr<iir::Expr>
IIRBuilder::make_reduce_over_neighbor_expr(op operation, std::shared_ptr<iir::Expr> const& rhs,
                                           std::shared_ptr<iir::Expr> const& init) {
  auto expr = std::make_shared<iir::ReductionOverNeighborExpr>(
      to_str(operation, {op::multiply, op::plus, op::minus, op::assign, op::divide}), rhs, init);
  expr->setID(si_->nextUID());
  return expr;
}
std::shared_ptr<iir::Expr> IIRBuilder::make_binary_expr(std::shared_ptr<iir::Expr> const& lhs,
                                                        std::shared_ptr<iir::Expr> const& rhs,
                                                        op operation) {
  auto binop = std::make_shared<iir::BinaryOperator>(
      lhs,
      to_str(operation,
             {op::multiply, op::plus, op::minus, op::divide, op::equal, op::not_equal, op::greater,
              op::less, op::greater_equal, op::less_equal, op::logical_and, op::logical_or}),
      rhs);
  binop->setID(si_->nextUID());
  return binop;
}
std::shared_ptr<iir::Expr> IIRBuilder::make_unary_expr(std::shared_ptr<iir::Expr> const& expr,
                                                       op operation) {
  auto ret = std::make_shared<iir::UnaryOperator>(
      expr, to_str(operation, {op::plus, op::minus, op::logical_not}));
  ret->setID(si_->nextUID());
  return ret;
}
std::shared_ptr<iir::Expr> IIRBuilder::make_assign_expr(std::shared_ptr<iir::Expr> const& lhs,
                                                        std::shared_ptr<iir::Expr> const& rhs,
                                                        op operation) {
  auto binop = std::make_shared<iir::AssignmentExpr>(
      lhs, rhs,
      to_str(operation, {op::assign, op::multiply, op::plus, op::minus, op::divide}) + "=");
  binop->setID(si_->nextUID());
  return binop;
}
int IIRBuilder::make_field(std::string const& name, field_type ft) {
  int ret = si_->getMetaData().addField(iir::FieldAccessType::FAT_APIField, name, as_array(ft));
  field_names_[ret] = name;
  field_ids_[name] = ret;
  return ret;
}
int IIRBuilder::make_localvar(std::string const& name) {
  // int ret = si_->getMetaData().addField(iir::FieldAccessType::FAT_LocalVariable, name,
  // as_array(field_type::ijk));
  int ret = si_->nextUID();
  field_names_[ret] = name;
  field_ids_[name] = ret;
  return ret;
}
std::shared_ptr<iir::Expr> IIRBuilder::at(int field_id, access_type access, Array3i extent) {
  auto expr = std::make_shared<iir::FieldAccessExpr>(field_names_[field_id], extent);
  expr->setID(si_->nextUID());

  si_->getMetaData().insertExprToAccessID(expr, field_id);

  if(access == access_type::r)
    read_extents_[expr.get()] = extent;
  else
    write_extents_[expr.get()] = extent;
  return expr;
}
std::shared_ptr<iir::Expr> IIRBuilder::at(int field_id, Array3i extent) {
  return at(field_id, access_type::r, extent);
}
std::unique_ptr<iir::StatementAccessesPair>
IIRBuilder::make_stmt(std::shared_ptr<iir::Expr>&& expr) {
  auto iir_stmt = std::make_shared<iir::ExprStmt>(std::move(expr));
  auto statement = std::make_shared<Statement>(iir_stmt, nullptr);
  return make_unique<iir::StatementAccessesPair>(statement);
}
std::unique_ptr<iir::StatementAccessesPair> IIRBuilder::declare_var(int var_id) {
  auto iir_stmt =
      std::make_shared<iir::VarDeclStmt>(Type{BuiltinTypeID::Float}, field_names_[var_id], 0, "=",
                                         std::vector<std::shared_ptr<Expr>>{});
  si_->getMetaData().addStmt(true, iir_stmt);
  auto statement = std::make_shared<Statement>(iir_stmt, nullptr);
  return make_unique<iir::StatementAccessesPair>(statement);
}

} // namespace iir
} // namespace dawn
