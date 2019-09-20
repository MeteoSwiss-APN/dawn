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
      computeAccesses(si_.get(), doMethod->getChildren());
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
  std::string op_str;
  switch(operation) {
  case op::multiply:
    op_str = "*";
    break;
  case op::plus:
    op_str = "+";
    break;
  case op::minus:
    op_str = "-";
    break;
  case op::assign:
    op_str = "";
    break;
  }
  auto expr = std::make_shared<iir::ReductionOverNeighborExpr>(op_str, rhs, init);
  expr->setID(si_->nextUID());
  return expr;
}
std::shared_ptr<iir::Expr> IIRBuilder::make_multiply_expr(std::shared_ptr<iir::Expr> const& lhs,
                                                          std::shared_ptr<iir::Expr> const& rhs) {
  auto binop = std::make_shared<iir::BinaryOperator>(lhs, "*", rhs);
  binop->setID(si_->nextUID());
  return binop;
}
std::shared_ptr<iir::Expr> IIRBuilder::make_assign_expr(std::shared_ptr<iir::Expr> const& lhs,
                                                        std::shared_ptr<iir::Expr> const& rhs,
                                                        op operation ) {
  std::string op_str;
  switch(operation) {
  case op::multiply:
    op_str = "*=";
    break;
  case op::plus:
    op_str = "+=";
    break;
  case op::minus:
    op_str = "-=";
    break;
  case op::assign:
    op_str = "=";
    break;
  default:
    DAWN_ASSERT(false);
  }
  auto binop = std::make_shared<iir::AssignmentExpr>(lhs, rhs, op_str);
  binop->setID(si_->nextUID());
  return binop;
}
int IIRBuilder::make_field(std::string const& name, field_type ft) {
  int ret = si_->getMetaData().addField(iir::FieldAccessType::FAT_APIField, name, as_array(ft));
  field_names_[ret] = name;
  field_ids_[name] = ret;
  return ret;
}
std::shared_ptr<iir::Expr> IIRBuilder::at(int field_id, access_type access ,
                                          Array3i extent ) {
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
  auto sap = make_unique<iir::StatementAccessesPair>(statement);
  auto accesses = std::make_shared<iir::Accesses>();

  struct ExtentAdder : public iir::ASTVisitorForwarding {
    void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
      if(b_.read_extents_.find(expr.get()) != b_.read_extents_.end()) {
        accesses_.addReadExtent(b_.field_ids_[expr->getName()], iir::Extents{0, 0, 0, 0, 0, 0});
        b_.read_extents_.erase(expr.get());
      }
      if(b_.write_extents_.find(expr.get()) != b_.write_extents_.end()) {
        accesses_.addWriteExtent(b_.field_ids_[expr->getName()], iir::Extents{0, 0, 0, 0, 0, 0});
        b_.write_extents_.erase(expr.get());
      }
      ASTVisitorForwarding::visit(expr);
    }
    ExtentAdder(iir::Accesses& accesses, IIRBuilder& b) : accesses_(accesses), b_(b) {}
    iir::Accesses& accesses_;
    IIRBuilder& b_;
  };
  ExtentAdder visitor{*accesses, *this};
  expr->accept(visitor);

  sap->setCallerAccesses(accesses);
  return sap;
}

} // namespace iir
} // namespace dawn
