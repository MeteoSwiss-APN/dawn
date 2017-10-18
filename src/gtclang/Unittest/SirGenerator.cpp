//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gtclang/Unittest/SirGenerator.h"
#include "dawn/SIR/ASTStringifier.h"
#include "dawn/Support/StringUtil.h"
#include <cstring>

#include <iostream>

namespace gtclang {

dawn::Type stringToType(const std::string& typestring) {
  if(typestring == "int") {
    return dawn::Type(dawn::BuiltinTypeID::Integer);
  }
  if(typestring == "float") {
    return dawn::Type(dawn::BuiltinTypeID::Float);
  }
  if(typestring == "auto") {
    return dawn::Type(dawn::BuiltinTypeID::Auto);
  }
  if(typestring == "bool") {
    return dawn::Type(dawn::BuiltinTypeID::Boolean);
  }
  dawn_unreachable("wrong type");
}

void FieldFinder::visit(const std::shared_ptr<dawn::FieldAccessExpr>& expr) {
  allFields_.push_back(std::make_shared<dawn::sir::Field>(expr->getName()));
  ASTVisitorForwarding::visit(expr);
}

void FieldFinder::visit(const std::shared_ptr<dawn::VerticalRegionDeclStmt>& stmt) {
  stmt->getVerticalRegion()->Ast->accept(*this);
}

std::shared_ptr<dawn::BlockStmt> block(const std::vector<std::shared_ptr<dawn::Stmt>>& statements) {
  return std::make_shared<dawn::BlockStmt>(statements);
}

std::shared_ptr<dawn::ExprStmt> expr(const std::shared_ptr<dawn::Expr>& expr) {
  return std::make_shared<dawn::ExprStmt>(expr);
}

std::shared_ptr<dawn::ReturnStmt> ret(const std::shared_ptr<dawn::Expr>& expr) {
  return std::make_shared<dawn::ReturnStmt>(expr);
}

std::shared_ptr<dawn::VarDeclStmt> vardec(const std::string& type, const std::string& name,
                                          const std::shared_ptr<dawn::Expr>& init, const char* op) {

  return vecdec(type, name, std::vector<std::shared_ptr<dawn::Expr>>({init}), 0, op);
}

std::shared_ptr<dawn::VarDeclStmt> vecdec(const std::string& type, const std::string& name,
                                          std::vector<std::shared_ptr<dawn::Expr>> initList,
                                          int dimension, const char* op) {
  auto realtype = stringToType(type);
  return std::make_shared<dawn::VarDeclStmt>(realtype, name, dimension, op, initList);
}

std::shared_ptr<dawn::VerticalRegionDeclStmt>
vrdec(const std::shared_ptr<dawn::sir::VerticalRegion>& verticalRegion) {
  return std::make_shared<dawn::VerticalRegionDeclStmt>(verticalRegion);
}

std::shared_ptr<dawn::StencilCallDeclStmt>
scdec(const std::shared_ptr<dawn::sir::StencilCall>& stencilCall) {
  return std::make_shared<dawn::StencilCallDeclStmt>(stencilCall);
}

std::shared_ptr<dawn::BoundaryConditionDeclStmt> bcdec(const std::__cxx11::string& callee) {
  return std::make_shared<dawn::BoundaryConditionDeclStmt>(callee);
}

std::shared_ptr<dawn::IfStmt> ifst(const std::shared_ptr<dawn::Stmt>& condExpr,
                                   const std::shared_ptr<dawn::Stmt>& thenStmt,
                                   const std::shared_ptr<dawn::Stmt>& elseStmt) {
  return std::make_shared<dawn::IfStmt>(condExpr, thenStmt, elseStmt);
}

std::shared_ptr<dawn::UnaryOperator> unop(const std::shared_ptr<dawn::Expr>& operand,
                                          const char* op) {
  return std::make_shared<dawn::UnaryOperator>(operand, op);
}

std::shared_ptr<dawn::BinaryOperator> binop(const std::shared_ptr<dawn::Expr>& left, const char* op,
                                            const std::shared_ptr<dawn::Expr>& right) {
  return std::make_shared<dawn::BinaryOperator>(left, op, right);
}

std::shared_ptr<dawn::AssignmentExpr> assign(const std::shared_ptr<dawn::Expr>& left,
                                             const std::shared_ptr<dawn::Expr>& right, const char* op) {
  return std::make_shared<dawn::AssignmentExpr>(left, right, op);
}

std::shared_ptr<dawn::TernaryOperator> ternop(const std::shared_ptr<dawn::Expr>& cond,
                                              const std::shared_ptr<dawn::Expr>& left,
                                              const std::shared_ptr<dawn::Expr>& right) {
  return std::make_shared<dawn::TernaryOperator>(cond, left, right);
}

std::shared_ptr<dawn::FunCallExpr> fcall(const std::__cxx11::string& callee) {
  return std::make_shared<dawn::FunCallExpr>(callee);
}

std::shared_ptr<dawn::StencilFunCallExpr> sfcall(const std::__cxx11::string& calee) {
  return std::make_shared<dawn::StencilFunCallExpr>(calee);
}

std::shared_ptr<dawn::StencilFunArgExpr> sfarg(int direction, int offset, int argumentIndex) {
  return std::make_shared<dawn::StencilFunArgExpr>(direction, offset, argumentIndex);
}

std::shared_ptr<dawn::VarAccessExpr> var(const std::__cxx11::string& name,
                                         std::shared_ptr<dawn::Expr> index) {
  return std::make_shared<dawn::VarAccessExpr>(name, index);
}

std::shared_ptr<dawn::FieldAccessExpr> field(const std::__cxx11::string& name, dawn::Array3i offset,
                                             dawn::Array3i argumentMap,
                                             dawn::Array3i argumentOffset, bool negateOffset) {
  return std::make_shared<dawn::FieldAccessExpr>(name, offset, argumentMap, argumentOffset,
                                                 negateOffset);
}

std::shared_ptr<dawn::LiteralAccessExpr> lit(const std::string& value,
                                             dawn::BuiltinTypeID builtinType) {
  return std::make_shared<dawn::LiteralAccessExpr>(value, builtinType);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::BlockStmt>& node) {
  std::cout << "Block Statement\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::VerticalRegionDeclStmt>& node) {
  std::cout << "VerticalRegionDeclStmt\n" << dawn::ASTStringifer::toString(node) << std::endl;
  node->getVerticalRegion()->Ast->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::StencilCallDeclStmt>& node) {
  std::cout << "StencilCallDeclStmt\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::BoundaryConditionDeclStmt>& node) {
  std::cout << "BoundaryConditionDeclStmt\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::IfStmt>& node) {
  std::cout << "IfStmt\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::UnaryOperator>& node) {
  std::cout << "UnaryOperator\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::BinaryOperator>& node) {
  std::cout << "BinaryOperator\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::AssignmentExpr>& node) {
  std::cout << "AssignmentExpr\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::TernaryOperator>& node) {
  std::cout << "TernaryOperator\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::FunCallExpr>& node) {
  std::cout << "FunCallExpr\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::StencilFunCallExpr>& node) {
  std::cout << "StencilFunCallExpr\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::StencilFunArgExpr>& node) {
  std::cout << "StencilFunArgExpr\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::VarAccessExpr>& node) {
  std::cout << "VarAccessExpr\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::FieldAccessExpr>& node) {
  std::cout << "FieldAccessExpr\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::LiteralAccessExpr>& node) {
  std::cout << "LiteralAccessExpr\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& s : node->getChildren())
    s->accept(*this);
}

void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::ExprStmt>& node) {
  std::cout << "ExprStmt\n" << dawn::ASTStringifer::toString(node) << std::endl;
  node->getExpr()->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::ReturnStmt>& node) {
  std::cout << "ReturnStmt\n" << dawn::ASTStringifer::toString(node) << std::endl;
  node->getExpr()->accept(*this);
}
void PrintAllExpressionTypes::visit(const std::shared_ptr<dawn::VarDeclStmt>& node) {
  std::cout << "VarDeclStmt\n" << dawn::ASTStringifer::toString(node) << std::endl;
  for(const auto& expr : node->getInitList())
    expr->accept(*this);
}

void ParsedString::argumentParsingImpl(ParsedString& p,
                                       const std::shared_ptr<dawn::Expr>& argument) {
  if(dawn::VarAccessExpr* expr = dawn::dyn_cast<dawn::VarAccessExpr>(argument.get())) {
    p.addVariable(expr->getName());
  } else if(dawn::FieldAccessExpr* expr = dawn::dyn_cast<dawn::FieldAccessExpr>(argument.get())) {
    p.addField(expr->getName());
  } else {
    dawn_unreachable("invalid expression");
  }
}

void ParsedString::dump() {
  std::cout << "function call: " << std::endl;
  std::cout << functionCall_ << std::endl;
  std::cout << "all fields: " << dawn::RangeToString()(fields_) << std::endl;
  std::cout << "all variables: " << dawn::RangeToString()(variables_) << std::endl;
}

} // namespace gtclang
