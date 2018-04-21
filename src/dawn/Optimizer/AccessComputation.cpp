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

#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/Accesses.h"
#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/Optimizer/StencilFunctionInstantiation.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTVisitor.h"
#include <iostream>
#include <stack>

namespace dawn {

namespace {

/// @brief Compute and fill the access map of the given statement
class AccessMapper : public ASTVisitor {
  StencilInstantiation* instantiation_;

  /// Keep track of the current statement access pair
  struct CurrentStatementAccessPair {
    CurrentStatementAccessPair(const std::shared_ptr<StatementAccessesPair>& pair)
        : Pair(pair), ChildIndex(0), IfCondExpr(nullptr) {}

    /// Reference to the pair we are currently working on.
    std::shared_ptr<StatementAccessesPair> Pair;

    /// Index of the child we are currently traversing. We need to keep track of this index because
    /// we fold the two block statements of an if/then/else block into a single vector of statements
    /// (i.e the childrens).
    int ChildIndex;

    /// Access of the condition of the if-statement (this will be added to all subsequent accesses)
    /// For example:
    ///   if(true)
    ///     in = 5.0;   // Accesses:  W:in, R:5.0, R:true   (true is part of the read-access)
    std::shared_ptr<Expr> IfCondExpr;
  };
  std::vector<std::unique_ptr<CurrentStatementAccessPair>> curStatementAccessPairStack_;

  /// List of caller and callee accesses. The first element is the primary accesses corresponding to
  /// the statement of the statement access pair which was passed to the constructor. All other
  /// elements are the accesses of the children of the top-level statement.
  std::vector<std::shared_ptr<Accesses>> callerAccessesList_;
  std::vector<std::shared_ptr<Accesses>> calleeAccessesList_;

  /// Reference to the stencil function we are currently inside (if any)
  std::shared_ptr<StencilFunctionInstantiation> stencilFun_;

  /// Reference to the current call of a stencil function if we are traversing an argument list
  struct StencilFunctionCallScope {
    StencilFunctionCallScope(std::shared_ptr<StencilFunctionInstantiation> functionInstantiation)
        : FunctionInstantiation(functionInstantiation), ArgumentIndex(0) {}

    std::shared_ptr<StencilFunctionInstantiation> FunctionInstantiation;
    int ArgumentIndex;
  };
  std::stack<std::unique_ptr<StencilFunctionCallScope>> stencilFunCalls_;

public:
  AccessMapper(StencilInstantiation* instantiation,
               const std::shared_ptr<StatementAccessesPair>& stmtAccessesPair,
               std::shared_ptr<StencilFunctionInstantiation> stencilFun = nullptr)
      : instantiation_(instantiation), stencilFun_(stencilFun) {
    curStatementAccessPairStack_.push_back(
        make_unique<CurrentStatementAccessPair>(stmtAccessesPair));
  }

  /// @brief Get the stencil function instantiation from the `StencilFunCallExpr`
  std::shared_ptr<StencilFunctionInstantiation>
  getStencilFunctionInstantiation(const std::shared_ptr<StencilFunCallExpr>& expr) {
    return (stencilFun_ ? stencilFun_->getStencilFunctionInstantiation(expr)
                        : instantiation_->getStencilFunctionInstantiation(expr));
  }

  /// @brief Get the AccessID from the Expr
  int getAccessIDFromExpr(const std::shared_ptr<Expr>& expr) {
    return stencilFun_ ? stencilFun_->getAccessIDFromExpr(expr)
                       : instantiation_->getAccessIDFromExpr(expr);
  }

  /// @brief Get the AccessID from the Stmt
  int getAccessIDFromStmt(const std::shared_ptr<Stmt>& stmt) {
    return stencilFun_ ? stencilFun_->getAccessIDFromStmt(stmt)
                       : instantiation_->getAccessIDFromStmt(stmt);
  }

  /// @brief Add a new access to the caller and callee and register it in the caller and callee
  /// accesses list. This will also add accesses to the children of the top-level statement access
  /// pair
  void appendNewAccesses() {
    curStatementAccessPairStack_.back()->Pair->setCallerAccesses(std::make_shared<Accesses>());
    callerAccessesList_.emplace_back(
        curStatementAccessPairStack_.back()->Pair->getCallerAccesses());

    if(stencilFun_) {
      curStatementAccessPairStack_.back()->Pair->setCalleeAccesses(std::make_shared<Accesses>());
      calleeAccessesList_.emplace_back(
          curStatementAccessPairStack_.back()->Pair->getCalleeAccesses());
    }

    // Add all accesses of all parent if-cond expressions
    for(const auto& pair : curStatementAccessPairStack_)
      if(pair->IfCondExpr)
        pair->IfCondExpr->accept(*this);
  }

  /// @brief Pop the last added child access from the caller and callee accesses list
  void removeLastChildAccesses() {

    // The top-level pair is never removed
    if(curStatementAccessPairStack_.size() <= 1)
      return;

    callerAccessesList_.pop_back();
    if(stencilFun_)
      calleeAccessesList_.pop_back();
  }

  /// @brief Add a write extent/offset to the caller and callee accesses
  /// @{
  void mergeWriteOffset(const std::shared_ptr<FieldAccessExpr>& field) {
    auto getOffset = [&](bool computeInitialOffset) {
      return (stencilFun_ ? stencilFun_->evalOffsetOfFieldAccessExpr(field, computeInitialOffset)
                          : field->getOffset());
    };

    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeWriteOffset(getAccessIDFromExpr(field), getOffset(true));

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeWriteOffset(getAccessIDFromExpr(field), getOffset(false));
  }

  void mergeWriteOffset(const std::shared_ptr<VarAccessExpr>& var) {
    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeWriteOffset(getAccessIDFromExpr(var), Array3i{{0, 0, 0}});

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeWriteOffset(getAccessIDFromExpr(var), Array3i{{0, 0, 0}});
  }

  void mergeWriteOffset(const std::shared_ptr<VarDeclStmt>& var) {
    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeWriteOffset(getAccessIDFromStmt(var), Array3i{{0, 0, 0}});

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeWriteOffset(getAccessIDFromStmt(var), Array3i{{0, 0, 0}});
  }

  void mergeWriteExtent(const std::shared_ptr<FieldAccessExpr>& field, const Extents& extent) {
    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeWriteExtent(getAccessIDFromExpr(field), extent);

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeWriteExtent(getAccessIDFromExpr(field), extent);
  }
  /// @}

  /// @brief Add a read offset/extent to the caller and callee accesses
  /// @{
  void mergeReadOffset(const std::shared_ptr<FieldAccessExpr>& field) {
    auto getOffset = [&](bool computeInitialOffset) {
      return (stencilFun_ ? stencilFun_->evalOffsetOfFieldAccessExpr(field, computeInitialOffset)
                          : field->getOffset());
    };

    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeReadOffset(getAccessIDFromExpr(field), getOffset(true));

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeReadOffset(getAccessIDFromExpr(field), getOffset(false));
  }

  void mergeReadOffset(const std::shared_ptr<VarAccessExpr>& var) {
    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeReadOffset(getAccessIDFromExpr(var), Array3i{{0, 0, 0}});

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeReadOffset(getAccessIDFromExpr(var), Array3i{{0, 0, 0}});
  }

  void mergeReadOffset(const std::shared_ptr<LiteralAccessExpr>& lit) {
    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeReadOffset(getAccessIDFromExpr(lit), Array3i{{0, 0, 0}});

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeReadOffset(getAccessIDFromExpr(lit), Array3i{{0, 0, 0}});
  }

  void mergeReadExtent(const std::shared_ptr<FieldAccessExpr>& field, const Extents& extent) {
    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeReadExtent(getAccessIDFromExpr(field), extent);

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeReadExtent(getAccessIDFromExpr(field), extent);
  }
  /// @}

  /// @brief Recursively merge the `extent` with all fields of the `curStencilFunCall` and apply
  /// them to the current *caller* accesses
  void mergeExtentWithAllFields(const Extents& extent,
                                std::shared_ptr<StencilFunctionInstantiation> curStencilFunCall,
                                std::set<int>& appliedAccessIDs) {
    for(const Field& field : curStencilFunCall->getCallerFields()) {
      int AccessID = field.getAccessID();

      if(appliedAccessIDs.count(AccessID))
        continue;

      if(curStencilFunCall->isProvidedByStencilFunctionCall(AccessID)) {
        // The field is provided by a stencil function, forward to the callee
        mergeExtentWithAllFields(
            extent, curStencilFunCall->getFunctionInstantiationOfArgField(
                        curStencilFunCall->getArgumentIndexFromCallerAccessID(AccessID)),
            appliedAccessIDs);

      } else {
        appliedAccessIDs.insert(AccessID);

        if(field.getIntend() == Field::IK_Input || field.getIntend() == Field::IK_InputOutput)
          for(auto& callerAccesses : callerAccessesList_)
            callerAccesses->addReadExtent(AccessID, extent);

        if(field.getIntend() == Field::IK_Output || field.getIntend() == Field::IK_InputOutput)
          for(auto& callerAccesses : callerAccessesList_)
            callerAccesses->addWriteExtent(AccessID, extent);
      }
    }
  }

  virtual void visit(const std::shared_ptr<BlockStmt>& stmt) override {
    // If we are inside the else block of an if-statement we need to continue iterating
    // the children as the if/then/else block of the if-statement has been collapsed into one single
    // vector of children
    if(!curStatementAccessPairStack_.back()->IfCondExpr)
      curStatementAccessPairStack_.back()->ChildIndex = 0;

    for(auto& s : stmt->getStatements()) {
      curStatementAccessPairStack_.push_back(make_unique<CurrentStatementAccessPair>(
          curStatementAccessPairStack_.back()
              ->Pair->getChildren()[curStatementAccessPairStack_.back()->ChildIndex]));

      // Process the statement
      s->accept(*this);

      curStatementAccessPairStack_.pop_back();
      curStatementAccessPairStack_.back()->ChildIndex++;
    }
  }

  virtual void visit(const std::shared_ptr<ExprStmt>& stmt) override {
    appendNewAccesses();
    stmt->getExpr()->accept(*this);
    removeLastChildAccesses();
  }

  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt) override {
    appendNewAccesses();
    stmt->getExpr()->accept(*this);
    removeLastChildAccesses();
  }

  void visit(const std::shared_ptr<IfStmt>& stmt) override {
    appendNewAccesses();
    stmt->getCondExpr()->accept(*this);

    curStatementAccessPairStack_.back()->IfCondExpr = stmt->getCondExpr();

    stmt->getThenStmt()->accept(*this);
    if(stmt->hasElse())
      stmt->getElseStmt()->accept(*this);

    curStatementAccessPairStack_.back()->IfCondExpr = nullptr;

    removeLastChildAccesses();
  }

  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    appendNewAccesses();

    // Declaration of variables are by defintion writes
    mergeWriteOffset(stmt);

    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);

    removeLastChildAccesses();
  }

  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override {
    DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
  }
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {
    DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
  }

  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override {
    DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
  }

  virtual void visit(const std::shared_ptr<FunCallExpr>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    StencilFunctionCallScope* previousStencilFunCallScope = nullptr;
    if(!stencilFunCalls_.empty())
      previousStencilFunCallScope = stencilFunCalls_.top().get();

    // Compute the accesses of the stencil function
    stencilFunCalls_.push(
        make_unique<StencilFunctionCallScope>(getStencilFunctionInstantiation(expr)));

    std::shared_ptr<StencilFunctionInstantiation> curStencilFunCall =
        stencilFunCalls_.top()->FunctionInstantiation;
    computeAccesses(curStencilFunCall, curStencilFunCall->getStatementAccessesPairs());

    // Compute the fields to get the IOPolicy of the arguments
    curStencilFunCall->update();

    // Traverse the Arguments
    for(const auto& arg : expr->getArguments())
      arg->accept(*this);

    // If the current stencil function is called within the argument list of another stencil
    // function, we need to merge the extents
    if(previousStencilFunCallScope) {
      auto& prevArgumentIndex = previousStencilFunCallScope->ArgumentIndex;
      auto& prevStencilFunCall = previousStencilFunCallScope->FunctionInstantiation;

      const Field& field = prevStencilFunCall->getCallerFieldFromArgumentIndex(prevArgumentIndex);
      DAWN_ASSERT(prevStencilFunCall->isProvidedByStencilFunctionCall(field.getAccessID()));

      // Add the extent of the field mapping to the stencil function to all fields
      std::set<int> appliedAccessIDs;
      mergeExtentWithAllFields(field.getExtents(), curStencilFunCall, appliedAccessIDs);

      prevArgumentIndex += 1;
    }

    // Done with the current stencil function call
    stencilFunCalls_.pop();
  }

  virtual void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override {
    stencilFunCalls_.top()->ArgumentIndex += 1;
  }

  void visit(const std::shared_ptr<AssignmentExpr>& expr) override {
    // LHS is a write, we resolve this manually as we only care about FieldAccessExpr and
    // VarAccessExpr. However, if we have an expression `a += 5` we need to register the access as
    // write and read!
    bool readAndWrite = StringRef(expr->getOp()) == "+=" || StringRef(expr->getOp()) == "-=" ||
                        StringRef(expr->getOp()) == "/=" || StringRef(expr->getOp()) == "*=" ||
                        StringRef(expr->getOp()) == "|=" || StringRef(expr->getOp()) == "&=";

    if(isa<FieldAccessExpr>(expr->getLeft().get())) {
      auto field = std::static_pointer_cast<FieldAccessExpr>(expr->getLeft());
      mergeWriteOffset(field);
      if(readAndWrite)
        mergeReadOffset(field);
    } else if(isa<VarAccessExpr>(expr->getLeft().get())) {
      auto var = std::static_pointer_cast<VarAccessExpr>(expr->getLeft());
      mergeWriteOffset(var);
      if(readAndWrite)
        mergeReadOffset(var);
    }

    // RHS are read accesses
    expr->getRight()->accept(*this);
  }

  void visit(const std::shared_ptr<UnaryOperator>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<BinaryOperator>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<TernaryOperator>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    // This is always a read access (writes are resolved in handling of the declaration of the
    // variable and in the assignment)
    mergeReadOffset(expr);

    // Resolve the index if this is an array access
    if(expr->isArrayAccess())
      expr->getIndex()->accept(*this);
  }

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    // Literals can, by defintion, only be read
    mergeReadOffset(expr);
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    if(!stencilFunCalls_.empty()) {

      std::shared_ptr<StencilFunctionInstantiation> functionInstantiation =
          stencilFunCalls_.top()->FunctionInstantiation;
      int& ArgumentIndex = stencilFunCalls_.top()->ArgumentIndex;

      // Get the extent of the field corresponding to the argument index
      const Field& field = functionInstantiation->getCallerFieldFromArgumentIndex(ArgumentIndex);

      if(field.getIntend() == Field::IK_Input || field.getIntend() == Field::IK_InputOutput)
        mergeReadExtent(expr, field.getExtents());

      if(field.getIntend() == Field::IK_Output || field.getIntend() == Field::IK_InputOutput)
        mergeWriteExtent(expr, field.getExtents());

      ArgumentIndex += 1;

    } else {
      // This is always a read access (writes are resolved in the the assignment)
      mergeReadOffset(expr);
    }
  }
};

} // anonymous namespace

void computeAccesses(StencilInstantiation* instantiation,
                     ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs) {
  for(const auto& statementAccessesPair : statementAccessesPairs) {
    DAWN_ASSERT(instantiation);
    AccessMapper mapper(instantiation, statementAccessesPair, nullptr);
    statementAccessesPair->getStatement()->ASTStmt->accept(mapper);
  }
}

void computeAccesses(std::shared_ptr<StencilFunctionInstantiation> stencilFunctionInstantiation,
                     ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs) {
  for(const auto& statementAccessesPair : statementAccessesPairs) {
    AccessMapper mapper(stencilFunctionInstantiation->getStencilInstantiation(),
                        statementAccessesPair, stencilFunctionInstantiation);
    statementAccessesPair->getStatement()->ASTStmt->accept(mapper);
  }
}

// anonymous namespace

} // namespace dawn
