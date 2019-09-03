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

#ifndef DAWN_SIR_ASTVISITOR_H
#define DAWN_SIR_ASTVISITOR_H

#include "dawn/AST/ASTVisitor.h"
#include "dawn/SIR/ASTData.h"
#include "dawn/SIR/ASTFwd.h"

namespace dawn {
namespace sir {

class ASTVisitor : virtual public ast::ASTVisitor<SIRASTData> {
public:
  using Base = ast::ASTVisitor<SIRASTData>;
  virtual ~ASTVisitor() {}
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) = 0;
};

class ASTVisitorNonConst : virtual public ast::ASTVisitorNonConst<SIRASTData> {
public:
  using Base = ast::ASTVisitorNonConst<SIRASTData>;
  virtual ~ASTVisitorNonConst() {}
  virtual void visit(std::shared_ptr<VerticalRegionDeclStmt> stmt) = 0;
};

class ASTVisitorForwarding : virtual public ast::ASTVisitorForwarding<SIRASTData> {
public:
  using Base = ast::ASTVisitorForwarding<SIRASTData>;
  virtual ~ASTVisitorForwarding() {}
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt);
};

class ASTVisitorPostOrder : virtual public ast::ASTVisitorPostOrder<SIRASTData> {
public:
  using Base = ast::ASTVisitorPostOrder<SIRASTData>;
  virtual ~ASTVisitorPostOrder() {}
  virtual std::shared_ptr<Stmt>
  visitAndReplace(std::shared_ptr<VerticalRegionDeclStmt> const& stmt);
  virtual bool preVisitNode(std::shared_ptr<VerticalRegionDeclStmt> const& stmt);
  virtual std::shared_ptr<Stmt> postVisitNode(std::shared_ptr<VerticalRegionDeclStmt> const& stmt);
};

class ASTVisitorForwardingNonConst : virtual public ast::ASTVisitorForwardingNonConst<SIRASTData> {
public:
  using Base = ast::ASTVisitorForwardingNonConst<SIRASTData>;
  virtual ~ASTVisitorForwardingNonConst() {}
  virtual void visit(std::shared_ptr<VerticalRegionDeclStmt> stmt);
};

class ASTVisitorDisabled : virtual public ast::ASTVisitorDisabled<SIRASTData> {
public:
  using Base = ast::ASTVisitorDisabled<SIRASTData>;
  virtual ~ASTVisitorDisabled() {}
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt);
};

} // namespace sir
} // namespace dawn

#endif
