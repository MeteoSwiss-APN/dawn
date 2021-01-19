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

#include "dawn/AST/ASTVisitor.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Support/Type.h"
#include <sstream>

namespace dawn {
namespace codegen {

/// @brief Abstract base class of all C++ code generation visitor
/// @ingroup codegen
class ASTCodeGenCXX : public ast::ASTVisitor, public NonCopyable {
protected:
  /// Indent of each statement
  int indent_;

  /// Current depth of the scope
  int scopeDepth_;

  /// Underlying stream
  std::stringstream ss_;

public:
  ASTCodeGenCXX();

  /// @name Statement implementation
  /// @{
  virtual void visit(const std::shared_ptr<ast::BlockStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ast::ExprStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ast::VarDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ast::IfStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ast::LoopStmt>& stmt) override{};
  /// @}

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) override{};
  virtual void visit(const std::shared_ptr<ast::UnaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<ast::BinaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<ast::AssignmentExpr>& expr) override;
  virtual void visit(const std::shared_ptr<ast::TernaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<ast::FunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<ast::LiteralAccessExpr>& expr) override;
  /// @}

  /// @brief Get the generated code and reset the internal string stream
  std::string getCodeAndResetStream();

  /// @brief Set initial indent of each statement
  void setIndent(int indent);

  /// @brief Mapping of VarDeclStmt and Var/FieldAccessExpr to their name
  /// @{
  virtual std::string getName(const std::shared_ptr<ast::Expr>& expr) const = 0;
  virtual std::string getName(const std::shared_ptr<ast::VarDeclStmt>& stmt) const = 0;
  /// @}

  /// @brief Convert builtin type to the corresponding C++ type
  static const char* builtinTypeIDToCXXType(const BuiltinTypeID& builtinTypeID, bool isAutoAllowed);
};

} // namespace codegen
} // namespace dawn
