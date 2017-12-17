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
#include "dawn/Optimizer/Interval.h"
#include "dawn/Support/StringUtil.h"
#include <stack>
#include <unordered_map>

namespace dawn {

class StencilInstantiation;
class StencilFunctionInstantiation;

/// @brief ASTVisitor to generate C++ gridtools code for the stencil and stencil function bodies
/// @ingroup codegen
class ASTCodeGenCXXNaiveStencilFnParamVisitor : public ASTVisitorDisabled, public NonCopyable {
protected:
  std::unordered_map<std::string, std::string> paramNameToType_;
  /// Underlying stream
  std::stringstream ss_;
  int firstParam_ = 0;

public:
  using Base = ASTVisitorDisabled;

  ASTCodeGenCXXNaiveStencilFnParamVisitor(
      std::unordered_map<std::string, std::string> paramNameToType);
  virtual ~ASTCodeGenCXXNaiveStencilFnParamVisitor();

  std::string getCodeAndResetStream();

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override;
  /// @}
};

} // namespace dawn
