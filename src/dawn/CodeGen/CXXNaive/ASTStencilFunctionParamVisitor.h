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

#ifndef DAWN_CODEGEN_CXXNAIVE_ASTSTENCILFUNCTIONPARAMVISITOR_H
#define DAWN_CODEGEN_CXXNAIVE_ASTSTENCILFUNCTIONPARAMVISITOR_H

#include "dawn/CodeGen/ASTCodeGenCXX.h"
#include "dawn/Optimizer/Interval.h"
#include "dawn/Support/StringUtil.h"
#include <stack>
#include <unordered_map>

namespace dawn {
class StencilInstantiation;
class StencilFunctionInstantiation;

namespace codegen {
namespace cxxnaive {

/// @brief ASTVisitor to generate C++ naive backend code for the parameters of the stencil function
/// calls
/// @ingroup cxxnaive
class ASTStencilFunctionParamVisitor : public ASTVisitorDisabled, public NonCopyable {
protected:
  const StencilInstantiation* instantiation_;
  const std::shared_ptr<StencilFunctionInstantiation>& currentFunction_;
  /// Underlying stream
  std::stringstream ss_;

public:
  using Base = ASTVisitorDisabled;

  ASTStencilFunctionParamVisitor(const std::shared_ptr<StencilFunctionInstantiation>& function,
                                 StencilInstantiation const* instantiation);
  virtual ~ASTStencilFunctionParamVisitor();

  std::string getCodeAndResetStream();

  std::string getName(const std::shared_ptr<Expr>& expr) const;

  int getAccessID(const std::shared_ptr<Expr>& expr) const;

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override;
  virtual void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override;
  /// @}
};

} // namespace cxxnaive
} // namespace codegen
} // namespace dawn

#endif
