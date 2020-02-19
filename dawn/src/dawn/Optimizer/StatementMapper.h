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

#include "dawn/IIR/ASTUtil.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include <stack>

namespace dawn {

//===------------------------------------------------------------------------------------------===//
//     StatementMapper
//===------------------------------------------------------------------------------------------===//
/// @brief Map the statements of the AST to a flat list of statements and assign AccessIDs to all
/// field, variable and literal accesses. In addition, stencil functions are instantiated.
class StatementMapper : public iir::ASTVisitor {

  /// @brief Representation of the current scope which keeps track of the binding of field and
  /// variable names
  struct Scope : public NonCopyable {
    Scope(iir::MultiStage& multiStage, const iir::Interval& interval,
          const iir::Stage::IterationSpace& iterationspace,
          const std::shared_ptr<iir::StencilFunctionInstantiation>& stencilFun)
        : multiStage_(multiStage), VerticalInterval(interval), iterationSpace(iterationspace),
          ScopeDepth(0), FunctionInstantiation(stencilFun), ArgumentIndex(0) {}
    /// Current Multistage that represents the VerticalRegionDeclStmt
    iir::MultiStage& multiStage_;

    // /// DoMethod containing the list of statements of the stencil function or stage
    // iir::DoMethod& doMethod_;

    /// The current interval
    const iir::Interval VerticalInterval;

    /// The current iterationspace
    iir::Stage::IterationSpace iterationSpace;

    /// Scope variable name to (global) AccessID
    std::unordered_map<std::string, int> LocalVarNameToAccessIDMap;

    /// Scope field name to (global) AccessID
    std::unordered_map<std::string, int> LocalFieldnameToAccessIDMap;

    /// Nesting of scopes
    int ScopeDepth;

    /// Reference to the current stencil function (may be NULL)
    std::shared_ptr<iir::StencilFunctionInstantiation> FunctionInstantiation;

    /// Counter of the parsed arguments
    int ArgumentIndex;

    /// During traversal of an argument list of a stencil function, this will hold the scope of
    /// the new stencil function
    std::stack<std::shared_ptr<Scope>> CandidateScopes;
  };

  iir::StencilInstantiation* instantiation_;
  iir::StencilMetaInformation& metadata_;
  OptimizerContext& context_;
  const std::vector<ast::StencilCall*>& stackTrace_;
  std::stack<std::shared_ptr<Scope>> scope_;
  bool initializedWithBlockStmt_ = false;

public:
  StatementMapper(
      iir::StencilInstantiation* instantiation, OptimizerContext& context,
      const std::vector<ast::StencilCall*>& stackTrace, iir::MultiStage& multiStage,
      const iir::Interval& interval, const iir::Stage::IterationSpace& iterationSpace,
      const std::unordered_map<std::string, int>& localFieldnameToAccessIDMap,
      const std::shared_ptr<iir::StencilFunctionInstantiation> stencilFunctionInstantiation);

  Scope* getCurrentCandidateScope();

  void appendNewStatement(const std::shared_ptr<iir::Stmt>& stmt);

  void visit(const std::shared_ptr<iir::BlockStmt>& stmt) override;

  void visit(const std::shared_ptr<iir::ExprStmt>& stmt) override;

  void visit(const std::shared_ptr<iir::ReturnStmt>& stmt) override;

  void visit(const std::shared_ptr<iir::IfStmt>& stmt) override;

  void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override;

  void visit(const std::shared_ptr<iir::UnaryOperator>& expr) override;

  void visit(const std::shared_ptr<iir::BinaryOperator>& expr) override;

  void visit(const std::shared_ptr<iir::TernaryOperator>& expr) override;

  void visit(const std::shared_ptr<iir::FunCallExpr>& expr) override;

  void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override;

  virtual void visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) override;

  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override;

  void visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) override;

  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override;

  void visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) override;
};

} // namespace dawn
