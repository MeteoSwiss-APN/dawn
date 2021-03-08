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
#include "dawn/IIR/Interval.h"
#include "dawn/Support/StringUtil.h"

#include <stack>
#include <unordered_map>

#include "LocToStringUtils.h"

namespace dawn {

namespace iir {
class StencilFunctionInstantiation;
class StencilMetaInformation;
} // namespace iir

namespace codegen {
namespace cudaico {

class FindReduceOverNeighborExpr : public ast::ASTVisitorForwardingNonConst {
  std::vector<std::shared_ptr<ast::ReductionOverNeighborExpr>> foundReductions_;

public:
  void visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) override {
    foundReductions_.push_back(expr);
    return;
  }
  bool hasReduceOverNeighborExpr() const { return !foundReductions_.empty(); }
  const std::vector<std::shared_ptr<ast::ReductionOverNeighborExpr>>&
  reduceOverNeighborExprs() const {
    return foundReductions_;
  }
};

/// @brief ASTVisitor to generate C++ naive code for the stencil and stencil function bodies
/// @ingroup cxxnaiveico
class ASTStencilBody : public ASTCodeGenCXX {
protected:
  const iir::StencilMetaInformation& metadata_;
  const Padding& padding_;

  // arg names for field access exprs
  std::string denseArgName_ = "loc";
  std::string sparseArgName_ = "loc";

  bool parentIsReduction_ = false;
  bool parentIsForLoop_ = false;
  std::optional<std::deque<int>> offsets_;

  bool firstPass_ = true;

  /// Nesting level of argument lists of stencil function *calls*
  int nestingOfStencilFunArgLists_;

  std::string makeIndexString(const std::shared_ptr<ast::FieldAccessExpr>& expr, std::string kiter);

public:
  using Base = ASTCodeGenCXX;
  using Base::visit;

  /// @brief constructor
  ASTStencilBody(const iir::StencilMetaInformation& metadata, const Padding& padding);

  virtual ~ASTStencilBody();

  void setFirstPass() { firstPass_ = true; };
  void setSecondPass() { firstPass_ = false; };

  /// @name Statement implementation
  /// @{
  void visit(const std::shared_ptr<ast::BlockStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::ReturnStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::LoopStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::VerticalRegionDeclStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::StencilCallDeclStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::BoundaryConditionDeclStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::IfStmt>& stmt) override;
  /// @}

  /// @name Expression implementation
  /// @{
  void visit(const std::shared_ptr<ast::FunCallExpr>& expr) override;
  void visit(const std::shared_ptr<ast::StencilFunCallExpr>& expr) override;
  void visit(const std::shared_ptr<ast::StencilFunArgExpr>& expr) override;
  void visit(const std::shared_ptr<ast::VarAccessExpr>& expr) override;
  void visit(const std::shared_ptr<ast::FieldAccessExpr>& expr) override;
  void visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) override;
  void visit(const std::shared_ptr<ast::AssignmentExpr>& expr) override;
  /// @}

  /// @brief Mapping of VarDeclStmt and Var/FieldAccessExpr to their name
  std::string getName(const std::shared_ptr<ast::Expr>& expr) const override;
  std::string getName(const std::shared_ptr<ast::VarDeclStmt>& stmt) const override;
};

} // namespace cudaico
} // namespace codegen
} // namespace dawn