//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_CODEGEN_ASTCODEGENCXX_H
#define GSL_CODEGEN_ASTCODEGENCXX_H

#include "gsl/SIR/ASTVisitor.h"
#include "gsl/Support/NonCopyable.h"
#include "gsl/Support/Type.h"
#include <sstream>

namespace gsl {

/// @brief Abstract base class of all C++ code generation visitor
/// @ingroup codegen
class ASTCodeGenCXX : public ASTVisitor, public NonCopyable {
protected:
  /// Indent of each statement
  int indent_;

  /// Current depth of the scope
  int scopeDepth_;

  /// Underlying stream
  std::stringstream ss_;

public:
  ASTCodeGenCXX();
  virtual ~ASTCodeGenCXX();

  /// @name Statement implementation
  /// @{
  virtual void visit(const std::shared_ptr<BlockStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ExprStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<IfStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt) override = 0;
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override = 0;
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override = 0;
  /// @}

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<UnaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<BinaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<AssignmentExpr>& expr) override;
  virtual void visit(const std::shared_ptr<TernaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<FunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override = 0;
  virtual void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override = 0;
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr) override = 0;
  virtual void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override = 0;
  /// @}

  /// @brief Get the generated code and reset the internal string stream
  std::string getCodeAndResetStream();

  /// @brief Set initial indent of each statement
  void setIndent(int indent);

  /// @brief Mapping of VarDeclStmt and Var/FieldAccessExpr to their name
  /// @{
  virtual const std::string& getName(const std::shared_ptr<Expr>& expr) const = 0;
  virtual const std::string& getName(const std::shared_ptr<Stmt>& stmt) const = 0;
  /// @}

  /// @brief Convert builtin type to the corresponding C++ type
  static const char* builtinTypeIDToCXXType(const BuiltinTypeID& builtinTypeID, bool isAutoAllowed);
};

} // namespace gsl

#endif
