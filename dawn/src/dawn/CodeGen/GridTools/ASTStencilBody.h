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
#include "dawn/IIR/Interval.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Support/StringUtil.h"
#include <stack>
#include <unordered_map>

namespace dawn {

namespace iir {
class StencilFunctionInstantiation;
}

namespace codegen {
namespace gt {

/// @brief ASTVisitor to generate C++ gridtools code for the stencil and stencil function bodies
/// @ingroup gt
class ASTStencilBody : public ASTCodeGenCXX {
protected:
  const iir::StencilMetaInformation& metadata_;
  const std::unordered_set<iir::IntervalProperties>& intervalProperties_;
  RangeToString offsetPrinter_;

  /// The stencil function we are currently generating or NULL
  std::shared_ptr<iir::StencilFunctionInstantiation> currentFunction_;

  /// Nesting level of argument lists of stencil function *calls*
  int nestingOfStencilFunArgLists_;

  bool triggerCallProc_ = false;

public:
  using Base = ASTCodeGenCXX;
  using Base::visit;

  ASTStencilBody(const iir::StencilMetaInformation& metadata,
                 const std::unordered_set<iir::IntervalProperties>& intervalProperties);
  virtual ~ASTStencilBody();

  /// @name Statement implementation
  /// @{
  virtual void visit(const std::shared_ptr<ast::ExprStmt>& stmt) override;
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

  /// @brief Set the current stencil function (can be NULL)
  void setCurrentStencilFunction(
      const std::shared_ptr<iir::StencilFunctionInstantiation>& currentFunction);

  /// @brief Mapping of VarDeclStmt and Var/FieldAccessExpr to their name
  std::string getName(const std::shared_ptr<ast::Expr>& expr) const override;
  std::string getName(const std::shared_ptr<ast::VarDeclStmt>& stmt) const override;
};

} // namespace gt
} // namespace codegen
} // namespace dawn
