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

#pragma once

#include "dawn/CodeGen/ASTCodeGenCXX.h"
#include "dawn/Support/StringUtil.h"
#include <stack>
#include <unordered_map>
#include <vector>

namespace dawn {
class StencilInstantiation;

namespace codegen {
namespace cxxnaive {

/// @brief ASTVisitor to generate C++ gridtools code for the stencil and stencil function bodies
/// @ingroup codegen
class ASTCodeGenCXXNaiveStencilDesc : public ASTCodeGenCXX {
protected:
  const dawn::StencilInstantiation* instantiation_;

  std::unordered_map<int, std::vector<std::string>> stencilIDToStencilNameMap_;

public:
  using Base = ASTCodeGenCXX;

  ASTCodeGenCXXNaiveStencilDesc(
      const dawn::StencilInstantiation* instantiation,
      const std::unordered_map<int, std::vector<std::string>>& stencilIDToStencilNameMap);

  virtual ~ASTCodeGenCXXNaiveStencilDesc();

  /// @name Statement implementation
  /// @{
  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override;
  /// @}

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override;
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override;
  /// @}

  const std::string& getName(const std::shared_ptr<Stmt>& stmt) const override;
  const std::string& getName(const std::shared_ptr<Expr>& expr) const override;
};

} // namespace cxxnaive
} // namespace codegen
} // namespace dawn
