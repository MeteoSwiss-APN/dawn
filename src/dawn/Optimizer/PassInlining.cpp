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
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/STLExtras.h"
#include <iostream>
#include <stack>
#include <unordered_map>
#include <vector>

namespace dawn {

namespace {

class Inliner;

static std::pair<bool, std::shared_ptr<Inliner>> tryInlineStencilFunction(
    PassInlining::InlineStrategyKind strategy,
    const std::shared_ptr<iir::StencilFunctionInstantiation>& stencilFunctioninstantiation,
    const std::unique_ptr<iir::StatementAccessesPair>& oldStmt,
    std::vector<std::unique_ptr<iir::StatementAccessesPair>>& newStmts, int AccessIDOfCaller,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation);

/// @brief Perform the inlining of a stencil-function
class Inliner : public ASTVisitor {
  PassInlining::InlineStrategyKind strategy_;
  const std::shared_ptr<iir::StencilFunctionInstantiation>& curStencilFunctioninstantiation_;
  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;
  iir::StencilMetaInformation& metadata_;

  /// The statement which we are currently processing in the `DetectInlineCandiates`
  const std::unique_ptr<iir::StatementAccessesPair>& oldStmtAccessesPair_;

  /// List of the new statements
  std::vector<std::unique_ptr<iir::StatementAccessesPair>>& newStmtAccessesPairs_;

  /// If a stencil function is called within the argument list of another stencil function, this
  /// stores the AccessID of the "temporary" storage we would need to store the return value of the
  /// stencil function (See StatementMapper::visit(const std::shared_ptr<StencilFunCallExpr>& expr)
  /// in StencilInstantiation.cpp). An AccessID of 0 indicates we are not in this scenario.
  int AccessIDOfCaller_;

  /// Nesting of the scopes
  int scopeDepth_;

  /// New expression which will be substitued for the `StencilFunCallExpr` (may be NULL)
  std::shared_ptr<Expr> newExpr_;

  /// Scope of the current argument list being parsed
  struct ArgListScope {
    ArgListScope(const std::shared_ptr<iir::StencilFunctionInstantiation>& function)
        : Function(function), ArgumentIndex(0) {}

    const std::shared_ptr<iir::StencilFunctionInstantiation>& Function;
    int ArgumentIndex;
  };

  std::stack<ArgListScope> argListScope_;
  std::stack<const std::unique_ptr<iir::StatementAccessesPair>*> currentStmtAccessesPair_;

public:
  Inliner(PassInlining::InlineStrategyKind strategy,
          const std::shared_ptr<iir::StencilFunctionInstantiation>& stencilFunctioninstantiation,
          const std::unique_ptr<iir::StatementAccessesPair>& oldStmtAccessesPair,
          std::vector<std::unique_ptr<iir::StatementAccessesPair>>& newStmtAccessesPairs,
          int AccessIDOfCaller,
          const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation)
      : strategy_(strategy), curStencilFunctioninstantiation_(stencilFunctioninstantiation),
        instantiation_(stencilInstantiation), metadata_(stencilInstantiation->getMetaData()),
        oldStmtAccessesPair_(oldStmtAccessesPair), newStmtAccessesPairs_(newStmtAccessesPairs),
        AccessIDOfCaller_(AccessIDOfCaller), scopeDepth_(0), newExpr_(nullptr) {}

  /// @brief Get the new expression which will be substitued for the `StencilFunCallExpr` of this
  /// `StencilFunctionInstantiation` (may be NULL)
  std::shared_ptr<Expr> getNewExpr() const { return newExpr_; }

  virtual void visit(const std::shared_ptr<BlockStmt>& stmt) override {
    scopeDepth_++;
    for(const auto& s : stmt->getStatements())
      s->accept(*this);
    scopeDepth_--;
  }

  void appendNewStatementAccessesPair(const std::shared_ptr<Stmt>& stmt) {
    if(scopeDepth_ == 1) {
      // The top-level block statement is collapsed thus we only insert at 1. Note that this works
      // because all AST have a block statement as root node.
      newStmtAccessesPairs_.emplace_back(make_unique<iir::StatementAccessesPair>(
          std::make_shared<Statement>(stmt, oldStmtAccessesPair_->getStatement()->StackTrace)));

      currentStmtAccessesPair_.push(&(newStmtAccessesPairs_.back()));

    } else if(scopeDepth_ > 1) {
      // We are inside a nested block statement, we add the stmt as a child of the parent statement
      (*currentStmtAccessesPair_.top())
          ->insertBlockStatement(make_unique<iir::StatementAccessesPair>(
              std::make_shared<Statement>(stmt, oldStmtAccessesPair_->getStatement()->StackTrace)));

      const std::unique_ptr<iir::StatementAccessesPair>& lp =
          (*(currentStmtAccessesPair_.top()))->getBlockStatements().back();

      currentStmtAccessesPair_.push(&lp);
    }
  }
  void removeLastChildStatementAccessesPair() {
    // The top-level pair is never removed
    if(currentStmtAccessesPair_.size() <= 1)
      return;

    currentStmtAccessesPair_.pop();
  }

  virtual void visit(const std::shared_ptr<ExprStmt>& stmt) override {
    appendNewStatementAccessesPair(stmt);
    stmt->getExpr()->accept(*this);
    removeLastChildStatementAccessesPair();
  }

  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt) override {
    DAWN_ASSERT_MSG(scopeDepth_ == 1, "cannot inline nested return statement!");

    // Instead of returning a value, we assign it to a local variable
    if(AccessIDOfCaller_ == 0) {
      // We are *not* called within an arugment list of a stencil function, meaning we can store the
      // return value in a local variable.
      int AccessID = instantiation_->nextUID();
      auto returnVarName = iir::InstantiationHelper::makeLocalVariablename(
          curStencilFunctioninstantiation_->getName(), AccessID);

      newExpr_ = std::make_shared<VarAccessExpr>(returnVarName);
      auto newStmt = std::make_shared<VarDeclStmt>(
          dawn::Type(BuiltinTypeID::Float, CVQualifier::Const), returnVarName, 0, "=",
          std::vector<std::shared_ptr<Expr>>{stmt->getExpr()});
      appendNewStatementAccessesPair(newStmt);

      // Register the variable
      metadata_.setAccessIDNamePair(AccessID, returnVarName);
      metadata_.mapStmtToAccessID(newStmt, AccessID);
      metadata_.mapExprToAccessID(newExpr_, AccessID);

    } else {
      // We are called within an arugment list of a stencil function, we thus need to store the
      // return value in temporary storage (we only land here if we do precomputations).
      auto returnFieldName = iir::InstantiationHelper::makeTemporaryFieldname(
          curStencilFunctioninstantiation_->getName(), AccessIDOfCaller_);

      newExpr_ = std::make_shared<FieldAccessExpr>(returnFieldName);
      auto newStmt =
          std::make_shared<ExprStmt>(std::make_shared<AssignmentExpr>(newExpr_, stmt->getExpr()));
      appendNewStatementAccessesPair(newStmt);

      // Promote the "temporary" storage we used to mock the argument to an actual temporary field
      metadata_.insertAccessOfType(iir::FieldAccessType::FAT_StencilTemporary, AccessIDOfCaller_,
                                   returnFieldName);
      metadata_.mapExprToAccessID(newExpr_, AccessIDOfCaller_);
    }

    // Resolve the actual expression of the return statement
    stmt->getExpr()->accept(*this);
  }

  void visit(const std::shared_ptr<IfStmt>& stmt) override {
    appendNewStatementAccessesPair(stmt);
    stmt->getCondExpr()->accept(*this);

    stmt->getThenStmt()->accept(*this);
    if(stmt->hasElse())
      stmt->getElseStmt()->accept(*this);

    removeLastChildStatementAccessesPair();
  }

  void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    int AccessID = curStencilFunctioninstantiation_->getAccessIDFromStmt(stmt);
    const std::string& name = curStencilFunctioninstantiation_->getFieldNameFromAccessID(AccessID);
    metadata_.setAccessIDNamePair(AccessID, name);
    metadata_.mapStmtToAccessID(stmt, AccessID);

    // Push back the statement and move on
    appendNewStatementAccessesPair(stmt);

    // Resolve the RHS
    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);

    removeLastChildStatementAccessesPair();
  }

  void visit(const std::shared_ptr<VerticalRegionDeclStmt>&) override {}
  void visit(const std::shared_ptr<StencilCallDeclStmt>&) override {}
  void visit(const std::shared_ptr<BoundaryConditionDeclStmt>&) override {}

  void visit(const std::shared_ptr<AssignmentExpr>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
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

  virtual void visit(const std::shared_ptr<FunCallExpr>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    // This is a nested stencil function call (i.e a stencil function call within the current
    // stencil function)
    std::shared_ptr<iir::StencilFunctionInstantiation> func =
        curStencilFunctioninstantiation_->getStencilFunctionInstantiation(expr);
    instantiation_->insertExprToStencilFunction(func);

    int AccessIDOfCaller = 0;
    if(!argListScope_.empty()) {
      int argIdx = argListScope_.top().ArgumentIndex;
      const std::shared_ptr<iir::StencilFunctionInstantiation>& curFunc =
          argListScope_.top().Function;

      // This stencil function is called within the argument list of another stencil function. Get
      // the AccessID of the "temporary" storage we used to mock the argument (this will be
      // converted to an actual temporary storage if we inline the function)
      AccessIDOfCaller = curFunc->getCallerAccessIDOfArgField(argIdx);
      DAWN_ASSERT(curFunc->isProvidedByStencilFunctionCall(AccessIDOfCaller));
    }

    // Resolve the arguments
    argListScope_.push(ArgListScope(func));
    for(const auto& arg : expr->getArguments())
      arg->accept(*this);
    argListScope_.pop();

    int oldSize = newStmtAccessesPairs_.size();
    // Try to inline the stencil-function
    auto inlineResult =
        tryInlineStencilFunction(strategy_, func, oldStmtAccessesPair_, newStmtAccessesPairs_,
                                 AccessIDOfCaller, instantiation_);

    // Compute the index of the statement of our current stencil-function call
    const int stmtIdxOfFunc = oldSize - 1;
    if(inlineResult.first) {
      if(func->hasReturn()) {
        std::shared_ptr<Inliner>& inliner = inlineResult.second;
        DAWN_ASSERT(inliner);
        DAWN_ASSERT(inliner->getNewExpr());

        // We need to change the current statement s.t instead of calling the stencil-function it
        // accesses the precomputed value. In addition the statement needs to be last again (we
        // push backed all the new statements). Hence, we need to insert an empty statement in the
        // back -> swap with our statement -> replace the expr in our statement and evict the empty
        // statement)
        newStmtAccessesPairs_.emplace_back(
            make_unique<iir::StatementAccessesPair>(std::make_shared<Statement>(nullptr, nullptr)));
        std::iter_swap(newStmtAccessesPairs_.begin() + stmtIdxOfFunc,
                       std::prev(newStmtAccessesPairs_.end()));

        replaceOldExprWithNewExprInStmt(
            newStmtAccessesPairs_[newStmtAccessesPairs_.size() - 1]->getStatement()->ASTStmt, expr,
            inliner->getNewExpr());
      }

      // Erase the statement of the original stencil function call. The statment is either empty
      // (in case it had a return value) or it just contains the function call which we inlined.
      newStmtAccessesPairs_.erase(newStmtAccessesPairs_.begin() + stmtIdxOfFunc);

      // Remove the function
      instantiation_->removeStencilFunctionInstantiation(expr, curStencilFunctioninstantiation_);

    } else {
      // Inlining failed, transfer ownership
      instantiation_->insertExprToStencilFunction(func);
    }

    if(!argListScope_.empty())
      argListScope_.top().ArgumentIndex++;
  }

  void visit(const std::shared_ptr<StencilFunArgExpr>&) override {
    if(!argListScope_.empty())
      argListScope_.top().ArgumentIndex++;
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {

    std::string callerName = metadata_.getFieldNameFromAccessID(
        curStencilFunctioninstantiation_->getAccessIDFromExpr(expr));
    expr->setName(callerName);

    metadata_.mapExprToAccessID(expr, curStencilFunctioninstantiation_->getAccessIDFromExpr(expr));
    if(expr->isArrayAccess())
      expr->getIndex()->accept(*this);
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {

    std::string callerName = metadata_.getFieldNameFromAccessID(
        curStencilFunctioninstantiation_->getAccessIDFromExpr(expr));
    expr->setName(callerName);

    metadata_.mapExprToAccessID(expr, curStencilFunctioninstantiation_->getAccessIDFromExpr(expr));

    // Set the fully evaluated offset as the new offset of the field. Note that this renders the
    // AST of the current stencil function incorrent which is why it needs to be removed!
    expr->setPureOffset(curStencilFunctioninstantiation_->evalOffsetOfFieldAccessExpr(expr, true));

    if(!argListScope_.empty())
      argListScope_.top().ArgumentIndex++;
  }

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    int AccessID = curStencilFunctioninstantiation_->getAccessIDFromExpr(expr);
    metadata_.insertAccessOfType(iir::FieldAccessType::FAT_Literal, AccessID, expr->getValue());
    metadata_.mapExprToAccessID(expr, AccessID);
  }
};

/// @brief Detect inline candidates
class DetectInlineCandiates : public ASTVisitorForwarding {
  PassInlining::InlineStrategyKind strategy_;
  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;

  /// The statement we are currently analyzing
  std::unique_ptr<iir::StatementAccessesPair> oldStmtAccessesPair_;

  /// If non-empty the `oldStmt` will be appended to `newStmts` with the given replacements
  std::unordered_map<std::shared_ptr<Expr>, std::shared_ptr<Expr>> replacmentOfOldStmtMap_;

  /// The new list of StatementAccessesPair which can serve as a replacement for `oldStmt`
  std::vector<std::unique_ptr<iir::StatementAccessesPair>> newStmtAccessesPairs_;

  /// If `true` we need to replace `oldStmt` with `newStmts`
  bool inlineCandiatesFound_;

  /// Scope of the current argument list being parsed
  struct ArgListScope {
    ArgListScope(const std::shared_ptr<iir::StencilFunctionInstantiation>& function)
        : Function(function), ArgumentIndex(0) {}

    const std::shared_ptr<iir::StencilFunctionInstantiation>& Function;
    int ArgumentIndex;
  };

  std::stack<ArgListScope> argListScope_;

public:
  using Base = ASTVisitorForwarding;

  DetectInlineCandiates(PassInlining::InlineStrategyKind strategy,
                        const std::shared_ptr<iir::StencilInstantiation>& instantiation)
      : strategy_(strategy), instantiation_(instantiation), inlineCandiatesFound_(false) {}

  /// @brief Process the given statement
  void processStatment(const std::unique_ptr<iir::StatementAccessesPair>& stmtAccesesPair) {
    // Reset the state
    inlineCandiatesFound_ = false;
    oldStmtAccessesPair_ = stmtAccesesPair->clone();
    newStmtAccessesPairs_.clear();

    // Detect the stencil functions suitable for inlining
    oldStmtAccessesPair_->getStatement()->ASTStmt->accept(*this);
  }

  /// @brief Atleast one inline candiate was found and the given `stmt` should be replaced with
  /// `getNewStatements`
  bool inlineCandiatesFound() const { return inlineCandiatesFound_; }

  /// @brief Get the newly computed statements which can be substituted for the given `stmt`
  ///
  /// Note that the accesses are not computed!
  std::vector<std::unique_ptr<iir::StatementAccessesPair>>& getNewStatementAccessesPairs() {
    if(!replacmentOfOldStmtMap_.empty()) {
      newStmtAccessesPairs_.push_back(oldStmtAccessesPair_->clone());

      for(const auto& oldNewPair : replacmentOfOldStmtMap_)
        replaceOldExprWithNewExprInStmt(
            newStmtAccessesPairs_[newStmtAccessesPairs_.size() - 1]->getStatement()->ASTStmt,
            oldNewPair.first, oldNewPair.second);

      // Clear the map in case someone would call getNewStatments multiple times
      replacmentOfOldStmtMap_.clear();
    }

    return newStmtAccessesPairs_;
  }

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    std::shared_ptr<iir::StencilFunctionInstantiation> func =
        instantiation_->getStencilFunctionInstantiation(expr);

    int AccessIDOfCaller = 0;
    if(!argListScope_.empty()) {
      int argIdx = argListScope_.top().ArgumentIndex;
      std::shared_ptr<iir::StencilFunctionInstantiation> curFunc = argListScope_.top().Function;

      // This stencil function is called within the argument list of another stencil function. Get
      // the AccessID of the "temporary" storage we used to mock the argument (this will be
      // converted to an actual temporary storage if we inline the function)
      AccessIDOfCaller = curFunc->getCallerAccessIDOfArgField(argIdx);
      DAWN_ASSERT(curFunc->isProvidedByStencilFunctionCall(AccessIDOfCaller));
    }

    argListScope_.push(ArgListScope(func));
    for(const auto& arg : expr->getArguments())
      arg->accept(*this);
    argListScope_.pop();

    auto inlineResult =
        tryInlineStencilFunction(strategy_, func, oldStmtAccessesPair_, newStmtAccessesPairs_,
                                 AccessIDOfCaller, instantiation_);

    inlineCandiatesFound_ |= inlineResult.first;
    if(inlineResult.first) {

      // Replace `StencilFunCallExpr` with an access to a storage or variable containing the result
      // of the stencil function call
      if(inlineResult.second->getNewExpr())
        replacmentOfOldStmtMap_.emplace(expr, inlineResult.second->getNewExpr());

      // Remove the stencil-function (`nullptr` means we don't have a nested stencil function)
      instantiation_->removeStencilFunctionInstantiation(expr, nullptr);
    }

    if(!argListScope_.empty())
      argListScope_.top().ArgumentIndex++;
  }

  void visit(const std::shared_ptr<StencilFunArgExpr>&) override {
    if(!argListScope_.empty())
      argListScope_.top().ArgumentIndex++;
  }

  void visit(const std::shared_ptr<FieldAccessExpr>&) override {
    if(!argListScope_.empty())
      argListScope_.top().ArgumentIndex++;
  }
};

/// @brief Decides if a stencil function is suitable for inlining and performs the inlining by
/// appending to `newStmts`
///
/// @param strategy          Inlining strategy to use (decides between computation on the fly and
///                          precomputation)
/// @param stencilFunc       The stencil function to be considered for inlining
/// @param oldStmt           The statement we are currently analyzing
/// @param newStmts          The list of new statements which serve as a replacement for `oldStmt`
/// @param AccessIDOfCaller  If the stencil function is called whithin the argument list of another
///                          stencil function this contains the AccessID of the "temporary" storage
///                          (otherwise it is 0)
/// @param stencilInstantiation    StencilInstantiation context from where the stencil function
/// candidate
///                          to inline is being called (null if the context is a stencil function
///                          instantiation)
/// @returns `true` if the stencil-function was inlined, `false` otherwise (the corresponding
/// `Inliner` instance (or NULL) is returned as well)
static std::pair<bool, std::shared_ptr<Inliner>> tryInlineStencilFunction(
    PassInlining::InlineStrategyKind strategy,
    const std::shared_ptr<iir::StencilFunctionInstantiation>& stencilFunc,
    const std::unique_ptr<iir::StatementAccessesPair>& oldStmtAccessesPair,
    std::vector<std::unique_ptr<iir::StatementAccessesPair>>& newStmtAccessesPairs,
    int AccessIDOfCaller, const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  // Function which do not return a value are *always* inlined. Function which do return a value
  // are only inlined if we favor precomputations.
  if(!stencilFunc->hasReturn() || strategy == PassInlining::IK_ComputationsOnTheFly) {
    auto inliner =
        std::make_shared<Inliner>(strategy, stencilFunc, oldStmtAccessesPair, newStmtAccessesPairs,
                                  AccessIDOfCaller, stencilInstantiation);
    stencilFunc->getAST()->accept(*inliner);
    return std::pair<bool, std::shared_ptr<Inliner>>(true, std::move(inliner));
  }
  return std::pair<bool, std::shared_ptr<Inliner>>(false, nullptr);
}

} // anonymous namespace

PassInlining::PassInlining(bool activate, InlineStrategyKind strategy)
    : Pass("PassInlining", true), activate_(activate), strategy_(strategy) {}

bool PassInlining::run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  DetectInlineCandiates inliner(strategy_, stencilInstantiation);

  if(!activate_)
    return true;

  // Iterate all statements (top -> bottom)
  for(const auto& stagePtr : iterateIIROver<iir::Stage>(*(stencilInstantiation->getIIR()))) {
    iir::Stage& stage = *stagePtr;
    for(const auto& doMethod : stage.getChildren()) {
      for(auto stmtAccIt = doMethod->childrenBegin(); stmtAccIt != doMethod->childrenEnd();
          ++stmtAccIt) {
        inliner.processStatment(*stmtAccIt);

        if(inliner.inlineCandiatesFound()) {
          auto& newStmtAccList = inliner.getNewStatementAccessesPairs();
          // Compute the accesses of the new statements
          computeAccesses(stencilInstantiation.get(), newStmtAccList);
          // Erase the old StatementAccessPair ...
          stmtAccIt = doMethod->childrenErase(stmtAccIt);

          // ... and insert the new ones
          // newStmtAccList will be cleared at the next for iteration, so it is safe to move the
          // elements here
          stmtAccIt =
              doMethod->insertChildren(stmtAccIt, std::make_move_iterator(newStmtAccList.begin()),
                                       std::make_move_iterator(newStmtAccList.end()));

          std::advance(stmtAccIt, newStmtAccList.size() - 1);
        }
      }
      doMethod->update(iir::NodeUpdateType::level);
    }

    stage.update(iir::NodeUpdateType::level);
  }
  for(const auto& MSPtr : iterateIIROver<iir::Stage>(*(stencilInstantiation->getIIR()))) {
    MSPtr->update(iir::NodeUpdateType::levelAndTreeAbove);
  }
  return true;
}

} // namespace dawn
