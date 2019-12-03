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

#include "dawn/CodeGen/ASTCodeGenCXX.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {

ASTCodeGenCXX::ASTCodeGenCXX() : indent_(0), scopeDepth_(0) {}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTCodeGenCXX::visit(const std::shared_ptr<iir::BlockStmt>& stmt) {
  scopeDepth_++;
  ss_ << std::string(indent_, ' ') << "{\n";

  indent_ += DAWN_PRINT_INDENT;
  auto indent = std::string(indent_, ' ');
  for(const auto& s : stmt->getStatements()) {
    ss_ << indent;
    s->accept(*this);
  }
  indent_ -= DAWN_PRINT_INDENT;

  ss_ << std::string(indent_, ' ') << "}\n";
  scopeDepth_--;
}

void ASTCodeGenCXX::visit(const std::shared_ptr<iir::ExprStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}

void ASTCodeGenCXX::visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  const auto& type = stmt->getType();
  if(type.isConst())
    ss_ << "const ";
  if(type.isVolatile())
    ss_ << "volatile ";

  if(type.isBuiltinType())
    ss_ << ASTCodeGenCXX::builtinTypeIDToCXXType(type.getBuiltinTypeID(), true);
  else
    ss_ << type.getName();
  ss_ << " " << getName(stmt);

  if(stmt->isArray())
    ss_ << "[" << stmt->getDimension() << "]";

  if(stmt->hasInit()) {
    ss_ << " " << stmt->getOp() << " ";
    if(!stmt->isArray())
      stmt->getInitList().front()->accept(*this);
    else {
      ss_ << "{";
      int numInit = stmt->getInitList().size();
      for(int i = 0; i < numInit; ++i) {
        stmt->getInitList()[i]->accept(*this);
        ss_ << ((i != (numInit - 1)) ? ", " : "");
      }
      ss_ << "}";
    }
  }
  ss_ << ";\n";
}

void ASTCodeGenCXX::visit(const std::shared_ptr<iir::IfStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  ss_ << "if(";
  stmt->getCondExpr()->accept(*this);
  ss_ << ")\n";

  stmt->getThenStmt()->accept(*this);
  if(stmt->hasElse()) {
    ss_ << std::string(indent_, ' ') << "else\n";
    stmt->getElseStmt()->accept(*this);
  }
}

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTCodeGenCXX::visit(const std::shared_ptr<iir::UnaryOperator>& expr) {
  ss_ << "(" << expr->getOp();
  expr->getOperand()->accept(*this);
  ss_ << ")";
}

void ASTCodeGenCXX::visit(const std::shared_ptr<iir::BinaryOperator>& expr) {
  ss_ << "(";
  expr->getLeft()->accept(*this);
  ss_ << " " << expr->getOp() << " ";
  expr->getRight()->accept(*this);
  ss_ << ")";
}

void ASTCodeGenCXX::visit(const std::shared_ptr<iir::AssignmentExpr>& expr) {
  expr->getLeft()->accept(*this);
  ss_ << " " << expr->getOp() << " ";
  expr->getRight()->accept(*this);
}

void ASTCodeGenCXX::visit(const std::shared_ptr<iir::TernaryOperator>& expr) {
  ss_ << "(";
  expr->getCondition()->accept(*this);
  ss_ << " " << expr->getOp() << " ";
  expr->getLeft()->accept(*this);
  ss_ << " " << expr->getSeperator() << " ";
  expr->getRight()->accept(*this);
  ss_ << ")";
}

void ASTCodeGenCXX::visit(const std::shared_ptr<iir::FunCallExpr>& expr) {
  ss_ << expr->getCallee() << "(";

  std::size_t numArgs = expr->getArguments().size();
  for(std::size_t i = 0; i < numArgs; ++i) {
    expr->getArguments()[i]->accept(*this);
    ss_ << (i == numArgs - 1 ? "" : ", ");
  }
  ss_ << ")";
}

void ASTCodeGenCXX::visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) {
  std::string type(ASTCodeGenCXX::builtinTypeIDToCXXType(expr->getBuiltinType(), false));
  ss_ << (type.empty() ? "" : "(" + type + ") ") << expr->getValue();
}

void ASTCodeGenCXX::setIndent(int indent) { indent_ = indent; }

std::string ASTCodeGenCXX::getCodeAndResetStream() {
  std::string str = ss_.str();
  codegen::clear(ss_);
  return str;
}

const char* ASTCodeGenCXX::builtinTypeIDToCXXType(const BuiltinTypeID& builtinTypeID,
                                                  bool isAutoAllowed) {
  switch(builtinTypeID) {
  case BuiltinTypeID::Invalid:
    return "";
  case BuiltinTypeID::Auto:
    return (isAutoAllowed ? "auto" : "");
  case BuiltinTypeID::Boolean:
    return "bool";
  case BuiltinTypeID::Float:
    return "::dawn::float_type";
  case BuiltinTypeID::Integer:
    return "int";
  default:
    dawn_unreachable("invalid builtin type");
  }
}

} // namespace codegen
} // namespace dawn
