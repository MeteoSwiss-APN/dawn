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

#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/ASTUtil.h"
#include <stack>

namespace dawn {

//===------------------------------------------------------------------------------------------===//
//     StatementMapper
//===------------------------------------------------------------------------------------------===//
/// @brief Map the statements of the AST to a flat list of statements and assign AccessIDs to all
/// field, variable and literal accesses. In addition, stencil functions are instantiated.
class StatementMapper : public ASTVisitor {

  /// @brief Representation of the current scope which keeps track of the binding of field and
  /// variable names
  struct Scope : public NonCopyable {
    Scope(iir::DoMethod& doMethod, const iir::Interval& interval,
          const std::shared_ptr<iir::StencilFunctionInstantiation>& stencilFun)
        : doMethod_(doMethod), VerticalInterval(interval), ScopeDepth(0),
          FunctionInstantiation(stencilFun), ArgumentIndex(0) {}

    /// DoMethod containing the list of statement/accesses pair of the stencil function or stage
    iir::DoMethod& doMethod_;

    /// Statement accesses pair pointing to the statement we are currently working on. This might
    /// not be the top-level statement which was passed to the constructor but rather a
    /// sub-statement (child) of the top-level statement if decend into nested block statements.
    std::stack<std::unique_ptr<iir::StatementAccessesPair> const*> CurentStmtAccessesPair;

    /// The current interval
    const iir::Interval VerticalInterval;

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

    /// Druring traversal of an argument list of a stencil function, this will hold the scope of
    /// the new stencil function
    std::stack<std::shared_ptr<Scope>> CandiateScopes;
  };

  const std::shared_ptr<SIR> sir_;
  iir::StencilInstantiation* instantiation_;
  iir::StencilMetaInformation& metadata_;
  std::shared_ptr<std::vector<sir::StencilCall*>> stackTrace_;
  std::stack<std::shared_ptr<Scope>> scope_;
  bool initializedWithBlockStmt_ = false;

public:
  StatementMapper(
      const std::shared_ptr<SIR>& fullSIR, iir::StencilInstantiation* instantiation,
      const std::shared_ptr<std::vector<sir::StencilCall*>>& stackTrace, iir::DoMethod& doMethod,
      const iir::Interval& interval,
      const std::unordered_map<std::string, int>& localFieldnameToAccessIDMap,
      const std::shared_ptr<iir::StencilFunctionInstantiation> stencilFunctionInstantiation);

  Scope* getCurrentCandidateScope();

  void appendNewStatementAccessesPair(const std::shared_ptr<Stmt>& stmt);

  void removeLastChildStatementAccessesPair();

  void visit(const std::shared_ptr<BlockStmt>& stmt) override;

  void visit(const std::shared_ptr<ExprStmt>& stmt) override;

  void visit(const std::shared_ptr<ReturnStmt>& stmt) override;

  void visit(const std::shared_ptr<IfStmt>& stmt) override;

  void visit(const std::shared_ptr<VarDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override;

  void visit(const std::shared_ptr<AssignmentExpr>& expr) override;

  void visit(const std::shared_ptr<UnaryOperator>& expr) override;

  void visit(const std::shared_ptr<BinaryOperator>& expr) override;

  void visit(const std::shared_ptr<TernaryOperator>& expr) override;

  void visit(const std::shared_ptr<FunCallExpr>& expr) override;

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override;

  virtual void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override;

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override;

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override;

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override;
};

} // namespace dawn
