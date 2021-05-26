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

#include "dawn/AST/ASTExpr.h"
#include "dawn/CodeGen/ASTCodeGenCXX.h"
#include "dawn/CodeGen/CXXNaive-ico/ASTStencilBody.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/IIR/Interval.h"
#include "dawn/Support/StringUtil.h"

#include <optional>
#include <sstream>
#include <stack>
#include <string>
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
  bool found_ = false;

public:
  void visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) override {
    found_ = true;
    return;
  }
  void visit(const std::shared_ptr<ast::BlockStmt>& stmt) override {
    // TODO do we need this override ?
    for(const auto& s : stmt->getStatements()) {
      s->accept(*this);
    }
  }
  bool hasReduceOverNeighborExpr() const { return found_; }
  FindReduceOverNeighborExpr() = default;
};

/// @brief ASTVisitor to generate C++ naive code for the stencil and stencil function bodies
/// @ingroup cxxnaiveico
class ASTStencilBody : public ASTCodeGenCXX {
protected:
  const iir::StencilMetaInformation& metadata_;

  // arg names for field access exprs
  std::string denseArgName_ = "loc";
  std::string sparseArgName_ = "loc";

  bool parentIsReduction_ = false;
  bool parentIsForLoop_ = false;
  bool genAtlasCompatCode_ = false;

  std::map<int, std::unique_ptr<ASTStencilBody>> reductionParser_;
  int recursiveIterNest_;

  /// Nesting level of argument lists of stencil function *calls*
  int nestingOfStencilFunArgLists_;

  std::string makeIndexString(const std::shared_ptr<ast::FieldAccessExpr>& expr, std::string kiter);
  bool hasIrregularPentagons(const std::vector<ast::LocationType>& chain);
  void evalNeighbourReductionLambda(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr);
  void generateNeighbourRedLoop(std::stringstream& ss) const;
  std::string pidx();
  std::string nbhLhsName(const std::shared_ptr<ast::Expr>& expr);
  std::string nbhIterStr();
  // symbol of the current neighbour index in a reduction loop
  std::string nbhIdxStr();
  // in case of nested reduction, this is the symbol of the index of parent reduction loop
  std::string nbhIdx_m1Str();

public:
  using Base = ASTCodeGenCXX;
  using Base::visit;

  /// @brief constructor

  ASTStencilBody(const iir::StencilMetaInformation& metadata, bool genAtlasCompatCode,
                 int recursiveIterNest = 0);

  virtual ~ASTStencilBody();

  /// @name Statement implementation
  /// @{
  void visit(const std::shared_ptr<ast::BlockStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::ReturnStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::LoopStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::VerticalRegionDeclStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::StencilCallDeclStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::BoundaryConditionDeclStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::IfStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::ExprStmt>& stmt) override;
  void visit(const std::shared_ptr<ast::VarDeclStmt>& stmt) override;
  /// @}

  /// @name Expression implementation
  /// @{
  void visit(const std::shared_ptr<ast::FunCallExpr>& expr) override;
  void visit(const std::shared_ptr<ast::StencilFunCallExpr>& expr) override;
  void visit(const std::shared_ptr<ast::StencilFunArgExpr>& expr) override;
  void visit(const std::shared_ptr<ast::VarAccessExpr>& expr) override;
  void visit(const std::shared_ptr<ast::LiteralAccessExpr>& expr) override;
  void visit(const std::shared_ptr<ast::UnaryOperator>& expr) override;
  void visit(const std::shared_ptr<ast::BinaryOperator>& expr) override;
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