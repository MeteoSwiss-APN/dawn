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
#include "dawn/Optimizer/OptimizerContext.h"

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
std::string toStr(Op operation, std::vector<Op> const& valid_ops) {
  DAWN_ASSERT(std::find(valid_ops.begin(), valid_ops.end(), operation) != valid_ops.end());
  switch(operation) {
  case Op::plus:
    return "+";
  case Op::minus:
    return "-";
  case Op::multiply:
    return "*";
  case Op::assign:
    return "";
  case Op::divide:
    return "/";
  case Op::equal:
    return "==";
  case Op::notEqual:
    return "!=";
  case Op::greater:
    return ">";
  case Op::less:
    return "<";
  case Op::greaterEqual:
    return ">=";
  case Op::lessEqual:
    return "<=";
  case Op::logicalAnd:
    return "&&";
  case Op::locigalOr:
    return "||";
  case Op::logicalNot:
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
std::shared_ptr<iir::Expr>
IIRBuilder::reduceOverNeighborExpr(Op operation, std::shared_ptr<iir::Expr>&& rhs,
                                   std::shared_ptr<iir::Expr>&& init,
                                   ast::Expr::LocationType rhs_location) {
  auto expr = std::make_shared<iir::ReductionOverNeighborExpr>(
      toStr(operation, {Op::multiply, Op::plus, Op::minus, Op::assign, Op::divide}), std::move(rhs),
      std::move(init), rhs_location);
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
      std::move(expr), toStr(operation, {Op::plus, Op::minus, Op::logicalNot}));
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
IIRBuilder::LocalVar IIRBuilder::localvar(std::string const& name, BuiltinTypeID type) {
  DAWN_ASSERT(si_);
  auto iirStmt = makeVarDeclStmt(Type{type}, name, 0, "=", std::vector<std::shared_ptr<Expr>>{});
  int id = si_->getMetaData().addStmt(true, iirStmt);
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

std::shared_ptr<iir::Stmt> IIRBuilder::declareVar(IIRBuilder::LocalVar& var) {
  DAWN_ASSERT(si_);
  DAWN_ASSERT(var.decl);
  return var.decl;
}

IIRBuilder::Field CartesianIIRBuilder::field(std::string const& name, FieldType ft) {
  DAWN_ASSERT(si_);
  auto fieldMaskArray = asArray(ft);
  sir::FieldDimension dimensions(
      ast::cartesian, {fieldMaskArray[0] == 1, fieldMaskArray[1] == 1, fieldMaskArray[2] == 1});
  int id = si_->getMetaData().addField(iir::FieldAccessType::APIField, name, dimensions);
  return {id, name};
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
                                                ast::Expr::LocationType location) {
  DAWN_ASSERT(si_);
  int id = si_->getMetaData().addField(iir::FieldAccessType::APIField, name,
                                       sir::FieldDimension(ast::cartesian, {true, true, true}),
                                       location);
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
