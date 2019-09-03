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

#include "dawn/AST/AST.h"

#include "dawn/Support/Printing.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Type.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>

namespace dawn {
namespace ast {

template <typename DataTraits>
StringVisitor<DataTraits>::StringVisitor(int initialIndent, bool newLines)
    : curIndent_(initialIndent), scopeDepth_(0), newLines_(newLines) {}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<BlockStmt<DataTraits>>& stmt) {
  scopeDepth_++;
  ss_ << std::string(curIndent_, ' ') << "{" << (newLines_ ? "\n" : "");

  curIndent_ += DAWN_PRINT_INDENT;

  auto indent = std::string(curIndent_, ' ');
  for(const auto& s : stmt->getStatements()) {
    ss_ << indent;
    s->accept(*this);
  }

  curIndent_ -= DAWN_PRINT_INDENT;

  ss_ << std::string(curIndent_, ' ') << "}\n";
  scopeDepth_--;
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<ExprStmt<DataTraits>>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(curIndent_, ' ');

  stmt->getExpr()->accept(*this);
  ss_ << ";" << (newLines_ ? "\n" : "");
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<ReturnStmt<DataTraits>>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(curIndent_, ' ');

  ss_ << "return ";
  stmt->getExpr()->accept(*this);
  ss_ << ";" << (newLines_ ? "\n" : "");
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<VarDeclStmt<DataTraits>>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(curIndent_, ' ');

  ss_ << stmt->getType() << " " << stmt->getName();
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
  ss_ << ";" << (newLines_ ? "\n" : "");
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(
    const std::shared_ptr<StencilCallDeclStmt<DataTraits>>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(curIndent_, ' ');
  ss_ << "stencil-call:";
  ss_ << stmt->getStencilCall()->Callee;
  ss_ << RangeToString(", ", "(", ")")(stmt->getStencilCall()->Args,
                                       [&](const std::string& fieldName) { return fieldName; });
  ss_ << ";" << (newLines_ ? "\n" : "");
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(
    const std::shared_ptr<BoundaryConditionDeclStmt<DataTraits>>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(curIndent_, ' ');
  ss_ << "boundary-condition:";
  ss_ << stmt->getFunctor();
  ss_ << RangeToString(", ", "(", ")")(stmt->getFields());
  ss_ << ";" << (newLines_ ? "\n" : "");
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<IfStmt<DataTraits>>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(curIndent_, ' ');
  ss_ << "if(";
  stmt->getCondExpr()->accept(*this);
  ss_ << ")\n";

  stmt->getThenStmt()->accept(*this);
  if(stmt->hasElse()) {
    ss_ << std::string(curIndent_, ' ') << "else\n";
    stmt->getElseStmt()->accept(*this);
  }
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<UnaryOperator<DataTraits>>& expr) {
  ss_ << "(";
  ss_ << expr->getOp();
  expr->getOperand()->accept(*this);
  ss_ << ")";
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<BinaryOperator<DataTraits>>& expr) {
  ss_ << "(";
  expr->getLeft()->accept(*this);
  ss_ << " " << expr->getOp() << " ";
  expr->getRight()->accept(*this);
  ss_ << ")";
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<AssignmentExpr<DataTraits>>& expr) {
  expr->getLeft()->accept(*this);
  ss_ << " " << expr->getOp() << " ";
  expr->getRight()->accept(*this);
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<TernaryOperator<DataTraits>>& expr) {
  ss_ << "(";
  expr->getCondition()->accept(*this);
  ss_ << " " << expr->getOp() << " ";
  expr->getLeft()->accept(*this);
  ss_ << " " << expr->getSeperator() << " ";
  expr->getRight()->accept(*this);
  ss_ << ")";
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<FunCallExpr<DataTraits>>& expr) {
  ss_ << "fun-call:" << expr->getCallee() << "(";
  for(std::size_t i = 0; i < expr->getArguments().size(); ++i) {
    expr->getArguments()[i]->accept(*this);
    ss_ << (i == (expr->getArguments().size() - 1) ? ")" : ", ");
  }
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<StencilFunCallExpr<DataTraits>>& expr) {
  ss_ << "stencil-fun-call:" << expr->getCallee() << "(";
  for(std::size_t i = 0; i < expr->getArguments().size(); ++i) {
    expr->getArguments()[i]->accept(*this);
    ss_ << (i == (expr->getArguments().size() - 1) ? ")" : ", ");
  }
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<StencilFunArgExpr<DataTraits>>& expr) {
  if(!expr->needsLazyEval()) {
    switch(expr->getDimension()) {
    case 0:
      ss_ << "i";
      break;
    case 1:
      ss_ << "j";
      break;
    case 2:
      ss_ << "k";
      break;
    default:
      dawn_unreachable("invalid dimension");
    }
  } else {
    ss_ << "arg(" << expr->getArgumentIndex() << ")";
  }
  if(expr->getOffset() != 0)
    ss_ << (expr->getOffset() > 0 ? "+" : "") << expr->getOffset();
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<VarAccessExpr<DataTraits>>& expr) {
  ss_ << expr->getName();
  if(expr->isArrayAccess()) {
    ss_ << "[";
    expr->getIndex()->accept(*this);
    ss_ << "]";
  }
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<FieldAccessExpr<DataTraits>>& expr) {
  if(!expr->hasArguments()) {
    ss_ << expr->getName() << RangeToString()(expr->getOffset());
  } else {
    ss_ << expr->getName() << "[";

    const auto& argMap = expr->getArgumentMap();
    const auto& argOffset = expr->getArgumentOffset();

    for(int i = 0; i < expr->getArgumentMap().size(); ++i) {
      if(argMap[i] >= 0) {
        ss_ << "arg(" << argMap[i] << ")";
        if(argOffset[i] != 0)
          ss_ << (argOffset[i] > 0 ? "+" : "") << argOffset[i];
      } else {
        ss_ << expr->getOffset()[i];
      }
      ss_ << (i == (expr->getArgumentMap().size() - 1) ? "]" : ", ");
    }
  }
}

template <typename DataTraits>
void StringVisitor<DataTraits>::visit(const std::shared_ptr<LiteralAccessExpr<DataTraits>>& expr) {
  ss_ << expr->getValue();
}

template <typename DataTraits, class Visitor>
std::string ASTStringifier<DataTraits, Visitor>::toString(const AST<DataTraits>& ast,
                                                          int initialIndent, bool newLines) {
  Visitor strVisitor(initialIndent, newLines);
  ast.accept(strVisitor);
  return strVisitor.toString();
}

template <typename DataTraits, class Visitor>
std::string
ASTStringifier<DataTraits, Visitor>::toString(const std::shared_ptr<Stmt<DataTraits>>& stmt,
                                              int initialIndent, bool newLines) {
  Visitor strVisitor(initialIndent, newLines);
  stmt->accept(strVisitor);
  return strVisitor.toString();
}

template <typename DataTraits, class Visitor>
std::string
ASTStringifier<DataTraits, Visitor>::toString(const std::shared_ptr<Expr<DataTraits>>& expr,
                                              int initialIndent, bool newLines) {
  Visitor strVisitor(initialIndent, newLines);
  expr->accept(strVisitor);
  return strVisitor.toString();
}

template <typename DataTraits, class Visitor>
std::ostream& operator<<(std::ostream& os, const AST<DataTraits>& ast) {
  return (os << ASTStringifier<DataTraits, Visitor>::toString(ast));
}

template <typename DataTraits, class Visitor>
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Stmt<DataTraits>>& expr) {
  return (os << ASTStringifier<DataTraits, Visitor>::toString(expr));
}

template <typename DataTraits, class Visitor>
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Expr<DataTraits>>& stmt) {
  return (os << ASTStringifier<DataTraits, Visitor>::toString(stmt));
}

} // namespace ast
} // namespace dawn
