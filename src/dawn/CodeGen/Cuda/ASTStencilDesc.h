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

#ifndef DAWN_CODEGEN_CUDA_ASTSTENCILDESC_H
#define DAWN_CODEGEN_CUDA_ASTSTENCILDESC_H

#include "dawn/CodeGen/ASTCodeGenCXX.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/Support/StringUtil.h"
#include <stack>
#include <unordered_map>
#include <vector>

namespace dawn {
namespace codegen {
namespace cuda {

/// @brief ASTVisitor to generate C++ naive bakend code for the control flow code of stencils
/// @ingroup cuda
class ASTStencilDesc : public ASTCodeGenCXX {
protected:
  const iir::StencilMetaInformation& metadata_;

  const CodeGenProperties& codeGenProperties_;

public:
  using Base = ASTCodeGenCXX;

  ASTStencilDesc(const iir::StencilMetaInformation& metadata,
                 const CodeGenProperties& CodeGenProperties);

  virtual ~ASTStencilDesc();

  /// @name Statement implementation
  /// @{
  virtual void visit(const std::shared_ptr<iir::ReturnStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) override;
  /// @}

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override;
  /// @}

  std::string getName(const std::shared_ptr<iir::Stmt>& stmt) const override;
  std::string getName(const std::shared_ptr<iir::Expr>& expr) const override;
};

} // namespace cuda
} // namespace codegen
} // namespace dawn

#endif
