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

namespace codegen {
namespace cxxnaive {

// @brief context of a stencil body
// (pure stencil or a stencil function)
enum class StencilContext { E_Stencil, E_StencilFunction };

/// @brief ASTVisitor to generate C++ naive code for the stencil and stencil function bodies
/// @ingroup cxxnaive
class ASTStencilBody : public ASTCodeGenCXX {
protected:
  const dawn::StencilInstantiation* instantiation_;
  RangeToString offsetPrinter_;

  /// The stencil function we are currently generating or NULL
  const dawn::StencilFunctionInstantiation* currentFunction_;
  // map of stencil (or stencil function) parameter types to names
  std::unordered_map<std::string, std::string> paramNameToType_;

  /// Nesting level of argument lists of stencil function *calls*
  int nestingOfStencilFunArgLists_;

  StencilContext stencilContext_;

  /**
   * @brief produces a string of (i,j,k) accesses for the C++ generated naive code,
   * from an array of offseted accesses
   */
  template <long unsigned N>
  std::array<std::string, N> ijkfyOffset(const std::array<int, N>& offsets,
                                         std::string accessName) {
    int n = -1;
    std::array<std::string, N> res;
    std::transform(offsets.begin(), offsets.end(), res.begin(), [&](int const& off) {
      ++n;
      std::array<std::string, 3> indices{"i+", "j+", "k+"};

      return ((n < 4) ? indices[n] : ""))) + std::to_string(off) +
             ((stencilContext_ == StencilContext::E_StencilFunction)
                  ? "+" + accessName + "_offsets[" + std::to_string(n) + "]"
                  : "");
    });
    return res;
  }

public:
  using Base = ASTCodeGenCXX;

  /// @brief constructor
  ASTStencilBody(const dawn::StencilInstantiation* stencilInstantiation,
                 std::unordered_map<std::string, std::string> paramNameToType,
                 StencilContext stencilContext);

  virtual ~ASTStencilBody();

  /// @name Statement implementation
  /// @{
  virtual void visit(const std::shared_ptr<BlockStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ExprStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<IfStmt>& stmt) override;
  /// @}

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<UnaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<BinaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<AssignmentExpr>& expr) override;
  virtual void visit(const std::shared_ptr<TernaryOperator>& expr) override;
  virtual void visit(const std::shared_ptr<FunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override;
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override;
  /// @}

  /// @brief Set the current stencil function (can be NULL)
  void setCurrentStencilFunction(const dawn::StencilFunctionInstantiation* currentFunction);

  /// @brief Mapping of VarDeclStmt and Var/FieldAccessExpr to their name
  const std::string& getName(const std::shared_ptr<Expr>& expr) const override;
  const std::string& getName(const std::shared_ptr<Stmt>& stmt) const override;
  int getAccessID(const std::shared_ptr<Expr>& expr) const;
};

} // namespace cxxnaive
} // namespace codegen
} // namespace dawn
