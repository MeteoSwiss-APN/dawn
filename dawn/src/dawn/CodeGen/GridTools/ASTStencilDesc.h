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

#ifndef DAWN_CODEGEN_GRIDTOOLS_ASTSTENCILDESC_H
#define DAWN_CODEGEN_GRIDTOOLS_ASTSTENCILDESC_H

#include "dawn/CodeGen/ASTCodeGenCXX.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/Support/StringUtil.h"
#include <stack>
#include <unordered_map>
#include <vector>

namespace dawn {
namespace codegen {
namespace gt {

/// @brief ASTVisitor to generate C++ gridtools code for the stencil and stencil function bodies
/// @ingroup gt
class ASTStencilDesc : public ASTCodeGenCXX {
protected:
  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;
  const iir::StencilMetaInformation& metadata_;

  /// StencilID to the name of the generated stencils for this ID
  const CodeGenProperties& codeGenProperties_;
  const std::unordered_map<int, std::string>& stencilIdToArguments_;

public:
  using Base = ASTCodeGenCXX;

  ASTStencilDesc(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                 const CodeGenProperties& codeGenProperties,
                 const std::unordered_map<int, std::string>& stencilIdToArguments);

  virtual ~ASTStencilDesc();

  /// @name Statement implementation
  /// @{
  virtual void visit(const std::shared_ptr<iir::BlockStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::ExprStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::ReturnStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::IfStmt>& stmt) override;
  /// @}

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::UnaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<iir::BinaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::TernaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<iir::FunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override;
  /// @}

  std::string getName(const std::shared_ptr<iir::VarDeclStmt>& stmt) const override;
  std::string getName(const std::shared_ptr<iir::Expr>& expr) const override;
};

} // namespace gt
} // namespace codegen
} // namespace dawn

#endif
