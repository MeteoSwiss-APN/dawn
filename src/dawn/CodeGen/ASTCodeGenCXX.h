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

#ifndef DAWN_CODEGEN_ASTCODEGENCXX_H
#define DAWN_CODEGEN_ASTCODEGENCXX_H

#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Support/Type.h"
#include <sstream>

namespace dawn {
namespace codegen {

/// @brief Abstract base class of all C++ code generation visitor
/// @ingroup codegen
class ASTCodeGenCXX : public iir::ASTVisitor, public NonCopyable {
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
  virtual void visit(const std::shared_ptr<iir::BlockStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::ExprStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::IfStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::ReturnStmt>& stmt) override = 0;
  virtual void visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) override = 0;
  virtual void visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) override = 0;
  /// @}

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<iir::UnaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<iir::BinaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::TernaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<iir::FunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override = 0;
  virtual void visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) override = 0;
  virtual void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override = 0;
  virtual void visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override = 0;
  /// @}

  /// @brief Get the generated code and reset the internal string stream
  std::string getCodeAndResetStream();

  /// @brief Set initial indent of each statement
  void setIndent(int indent);

  /// @brief Mapping of VarDeclStmt and Var/FieldAccessExpr to their name
  /// @{
  virtual std::string getName(const std::shared_ptr<iir::Expr>& expr) const = 0;
  virtual std::string getName(const std::shared_ptr<iir::Stmt>& stmt) const = 0;
  /// @}

  /// @brief Convert builtin type to the corresponding C++ type
  static const char* builtinTypeIDToCXXType(const BuiltinTypeID& builtinTypeID, bool isAutoAllowed);
};

} // namespace codegen
} // namespace dawn

#endif
