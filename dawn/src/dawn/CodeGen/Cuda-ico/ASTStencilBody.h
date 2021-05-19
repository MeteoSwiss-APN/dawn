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

using MergeGroupMap =
    std::map<int, std::vector<std::vector<std::shared_ptr<ast::ReductionOverNeighborExpr>>>>;

class FindReduceOverNeighborExpr : public ast::ASTVisitorForwardingNonConst {
  std::vector<std::shared_ptr<ast::ReductionOverNeighborExpr>> foundReductions_;
  std::map<int, int> reductionToBlock_;
  int currentBlock_ = -1;

public:
  void visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) override {
    foundReductions_.push_back(expr);
    reductionToBlock_.insert({expr->getID(), currentBlock_});
    return;
  }
  void visit(const std::shared_ptr<ast::BlockStmt>& stmt) override {
    currentBlock_ = stmt->getID();
    for(const auto& s : stmt->getStatements()) {
      s->accept(*this);
    }
  }
  bool hasReduceOverNeighborExpr() const { return !foundReductions_.empty(); }
  const std::vector<std::shared_ptr<ast::ReductionOverNeighborExpr>>&
  reduceOverNeighborExprs() const {
    return foundReductions_;
  }
  int getBlockIDofReduction(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) const {
    return reductionToBlock_.at(expr->getID());
  }
  FindReduceOverNeighborExpr() = default;
  FindReduceOverNeighborExpr(int currentBlock) : currentBlock_(currentBlock){};
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
  int parentReductionID_ = -1;
  bool parentIsForLoop_ = false;
  std::optional<std::deque<int>> offsets_;

  bool firstPass_ = true;

  int currentBlock_ = -1;

  std::map<int, std::stringstream> reductionMap_;
  std::optional<MergeGroupMap> blockToMergeGroupMap_ = std::nullopt;

  /// Nesting level of argument lists of stencil function *calls*
  int nestingOfStencilFunArgLists_;

  std::string makeIndexString(const std::shared_ptr<ast::FieldAccessExpr>& expr, std::string kiter);
  bool hasIrregularPentagons(const std::vector<ast::LocationType>& chain);

public:
  using Base = ASTCodeGenCXX;
  using Base::visit;

  /// @brief constructor
  ASTStencilBody(const iir::StencilMetaInformation& metadata);

  virtual ~ASTStencilBody();

  void setFirstPass() { firstPass_ = true; };
  void setSecondPass() { firstPass_ = false; };
  void setBlockID(int currentBlock) { currentBlock_ = currentBlock; }
  void setBlockToMergeGroupMap(std::optional<MergeGroupMap> blockToMergeGroupMap) {
    blockToMergeGroupMap_ = blockToMergeGroupMap;
  }

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