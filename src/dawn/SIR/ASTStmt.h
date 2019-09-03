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

#ifndef DAWN_SIR_ASTSTMT_H
#define DAWN_SIR_ASTSTMT_H

#include "dawn/AST/ASTStmt.h"
#include "dawn/SIR/ASTData.h"

namespace dawn {
namespace sir {

using AST = ast::AST<SIRASTData>;
class ASTVisitor;
class ASTVisitorNonConst;
class ASTVisitorPostOrder;

class VerticalRegion;

using Stmt = ast::Stmt<SIRASTData>;
using BlockStmt = ast::BlockStmt<SIRASTData>;
using ExprStmt = ast::ExprStmt<SIRASTData>;
using ReturnStmt = ast::ReturnStmt<SIRASTData>;
using VarDeclStmt = ast::VarDeclStmt<SIRASTData>;
using StencilCallDeclStmt = ast::StencilCallDeclStmt<SIRASTData>;
using BoundaryConditionDeclStmt = ast::BoundaryConditionDeclStmt<SIRASTData>;
using IfStmt = ast::IfStmt<SIRASTData>;

//===------------------------------------------------------------------------------------------===//
//     VerticalRegionDeclStmt
//===------------------------------------------------------------------------------------------===//

/// @brief This represents a declaration of a VerticalRegion
/// @ingroup sir
class VerticalRegionDeclStmt : public SIRASTData::VerticalRegionDeclStmt, public Stmt {
  std::shared_ptr<AST> ast_; ///< AST of the region

public:
  /// @name Constructor & Destructor
  /// @{
  VerticalRegionDeclStmt(const std::shared_ptr<AST>& ast,
                         const std::shared_ptr<VerticalRegion>& verticalRegion,
                         SourceLocation loc = SourceLocation());
  VerticalRegionDeclStmt(const VerticalRegionDeclStmt& stmt);
  VerticalRegionDeclStmt& operator=(VerticalRegionDeclStmt stmt);
  virtual ~VerticalRegionDeclStmt();
  /// @}

  const std::shared_ptr<AST>& getAST() const { return ast_; }
  std::shared_ptr<AST>& getAST() { return ast_; }

  virtual bool isStencilDesc() const override { return true; }
  virtual std::shared_ptr<Stmt> clone() const override;
  virtual bool equals(const Stmt* other) const override;
  static bool classof(const Stmt* stmt) { return stmt->getKind() == SK_VerticalRegionDeclStmt; }
  virtual void accept(ast::ASTVisitor<SIRASTData>& visitor) override;
  virtual void accept(ast::ASTVisitorNonConst<SIRASTData>& visitor) override;
  virtual std::shared_ptr<Stmt>
  acceptAndReplace(ast::ASTVisitorPostOrder<SIRASTData>& visitor) override;
};

} // namespace sir
} // namespace dawn

#endif
