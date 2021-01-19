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
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/Support/StringUtil.h"
#include <stack>
#include <unordered_map>
#include <vector>

namespace dawn {
namespace iir {
class StencilMetaInformation;
}

namespace codegen {
namespace cxxnaive {

/// @brief ASTVisitor to generate C++ naive bakend code for the control flow code of stencils
/// @ingroup cxxnaive
class ASTStencilDesc : public ASTCodeGenCXX {
protected:
  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;
  const iir::StencilMetaInformation& metadata_;

  const CodeGenProperties& codeGenProperties_;

public:
  using Base = ASTCodeGenCXX;
  using Base::visit;

  ASTStencilDesc(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                 const CodeGenProperties& CodeGenProperties);

  virtual ~ASTStencilDesc();

  /// @name Statement implementation
  /// @{
  virtual void visit(const std::shared_ptr<ast::ReturnStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ast::VerticalRegionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ast::StencilCallDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ast::BoundaryConditionDeclStmt>& stmt) override;
  /// @}

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<ast::StencilFunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<ast::StencilFunArgExpr>& expr) override;
  virtual void visit(const std::shared_ptr<ast::VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<ast::FieldAccessExpr>& expr) override;
  /// @}

  std::string getName(const std::shared_ptr<ast::VarDeclStmt>& stmt) const override;
  std::string getName(const std::shared_ptr<ast::Expr>& expr) const override;
};

} // namespace cxxnaive
} // namespace codegen
} // namespace dawn
