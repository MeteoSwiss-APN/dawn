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

#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Printing.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>
#include <sstream>

namespace dawn {

namespace {

/// @brief Dump AST to string
class StringVisitor : public ASTVisitor {
  std::stringstream ss_;
  int curIndent_;
  int scopeDepth_;
  bool newLines_;

public:
  StringVisitor(int initialIndent, bool newLines)
      : curIndent_(initialIndent), scopeDepth_(0), newLines_(newLines) {}

  void visit(const std::shared_ptr<BlockStmt>& stmt) override {
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

  void visit(const std::shared_ptr<ExprStmt>& stmt) override {
    if(scopeDepth_ == 0)
      ss_ << std::string(curIndent_, ' ');

    stmt->getExpr()->accept(*this);
    ss_ << ";" << (newLines_ ? "\n" : "");
  }

  void visit(const std::shared_ptr<ReturnStmt>& stmt) override {
    if(scopeDepth_ == 0)
      ss_ << std::string(curIndent_, ' ');

    ss_ << "return ";
    stmt->getExpr()->accept(*this);
    ss_ << ";" << (newLines_ ? "\n" : "");
  }

  void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
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

  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override {
    if(scopeDepth_ == 0)
      ss_ << std::string(curIndent_, ' ');

    ss_ << "vertical-region : ";
    ss_ << *stmt->getVerticalRegion()->VerticalInterval.get();
    ss_ << " ["
        << (stmt->getVerticalRegion()->LoopOrder == sir::VerticalRegion::LK_Forward ? "forward"
                                                                                    : "backward")
        << "]\n";
    ss_ << ASTStringifer::toString(*stmt->getVerticalRegion()->Ast, curIndent_);
  }

  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {
    if(scopeDepth_ == 0)
      ss_ << std::string(curIndent_, ' ');
    ss_ << "stencil-call:";
    ss_ << stmt->getStencilCall()->Callee;
    ss_ << RangeToString(", ", "(", ")")(
        stmt->getStencilCall()->Args,
        [&](const std::shared_ptr<sir::Field>& field) { return field->Name; });
    ss_ << ";" << (newLines_ ? "\n" : "");
  }

  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override {
    if(scopeDepth_ == 0)
      ss_ << std::string(curIndent_, ' ');
    ss_ << "boundary-condition:";
    ss_ << stmt->getFunctor();
    ss_ << RangeToString(", ", "(", ")")(
        stmt->getFields(), [&](const std::shared_ptr<sir::Field>& field) { return field->Name; });
    ss_ << ";" << (newLines_ ? "\n" : "");
  }

  void visit(const std::shared_ptr<IfStmt>& stmt) override {
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

  void visit(const std::shared_ptr<UnaryOperator>& expr) override {
    ss_ << "(";
    ss_ << expr->getOp();
    expr->getOperand()->accept(*this);
    ss_ << ")";
  }

  void visit(const std::shared_ptr<BinaryOperator>& expr) override {
    ss_ << "(";
    expr->getLeft()->accept(*this);
    ss_ << " " << expr->getOp() << " ";
    expr->getRight()->accept(*this);
    ss_ << ")";
  }

  void visit(const std::shared_ptr<AssignmentExpr>& expr) override {
    expr->getLeft()->accept(*this);
    ss_ << " " << expr->getOp() << " ";
    expr->getRight()->accept(*this);
  }

  void visit(const std::shared_ptr<TernaryOperator>& expr) override {
    ss_ << "(";
    expr->getCondition()->accept(*this);
    ss_ << " " << expr->getOp() << " ";
    expr->getLeft()->accept(*this);
    ss_ << " " << expr->getSeperator() << " ";
    expr->getRight()->accept(*this);
    ss_ << ")";
  }

  void visit(const std::shared_ptr<FunCallExpr>& expr) override {
    ss_ << "fun-call:" << expr->getCallee() << "(";
    for(std::size_t i = 0; i < expr->getArguments().size(); ++i) {
      expr->getArguments()[i]->accept(*this);
      ss_ << (i == (expr->getArguments().size() - 1) ? ")" : ", ");
    }
  }

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    ss_ << "stencil-fun-call:" << expr->getCallee() << "(";
    for(std::size_t i = 0; i < expr->getArguments().size(); ++i) {
      expr->getArguments()[i]->accept(*this);
      ss_ << (i == (expr->getArguments().size() - 1) ? ")" : ", ");
    }
  }

  void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override {
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

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    ss_ << expr->getName();
    if(expr->isArrayAccess()) {
      ss_ << "[";
      expr->getIndex()->accept(*this);
      ss_ << "]";
    }
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
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

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override { ss_ << expr->getValue(); }

  std::string toString() const { return ss_.str(); }
};

} // namespace

std::string ASTStringifer::toString(const AST& ast, int initialIndent, bool newLines) {
  StringVisitor strVisitor(initialIndent, newLines);
  ast.accept(strVisitor);
  return strVisitor.toString();
}
std::string ASTStringifer::toString(const std::shared_ptr<Stmt>& stmt, int initialIndent,
                                    bool newLines) {
  StringVisitor strVisitor(initialIndent, newLines);
  stmt->accept(strVisitor);
  return strVisitor.toString();
}

std::string ASTStringifer::toString(const std::shared_ptr<Expr>& expr, int initialIndent,
                                    bool newLines) {
  StringVisitor strVisitor(initialIndent, newLines);
  expr->accept(strVisitor);
  return strVisitor.toString();
}

std::ostream& operator<<(std::ostream& os, const AST& ast) {
  return (os << ASTStringifer::toString(ast));
}
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Stmt>& expr) {
  return (os << ASTStringifer::toString(expr));
}
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Expr>& stmt) {
  return (os << ASTStringifer::toString(stmt));
}

} // namespace dawn
