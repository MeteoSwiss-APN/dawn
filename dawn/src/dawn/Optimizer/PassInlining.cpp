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
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTUtil.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/AccessComputation.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/STLExtras.h"

#include <stack>
#include <unordered_map>
#include <vector>

namespace dawn {

namespace {

class Inliner;

static std::pair<bool, std::shared_ptr<Inliner>> tryInlineStencilFunction(
    PassInlining::InlineStrategy strategy,
    const std::shared_ptr<iir::StencilFunctionInstantiation>& stencilFunctioninstantiation,
    const std::shared_ptr<iir::Stmt>& oldStmt, std::vector<std::shared_ptr<iir::Stmt>>& newStmts,
    int AccessIDOfCaller, const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation);

/// @brief Perform the inlining of a stencil-function
class Inliner : public iir::ASTVisitor {
  PassInlining::InlineStrategy strategy_;
  const std::shared_ptr<iir::StencilFunctionInstantiation>& curStencilFunctioninstantiation_;
  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;
  iir::StencilMetaInformation& metadata_;

  /// The statement which we are currently processing in the `DetectInlineCandiates`
  const std::shared_ptr<iir::Stmt>& oldStmt_;

  /// List of the new statements
  std::vector<std::shared_ptr<iir::Stmt>>& newStmts_;

  /// If a stencil function is called within the argument list of another stencil function, this
  /// stores the AccessID of the "temporary" storage we would need to store the return value of the
  /// stencil function (See StatementMapper::visit(const std::shared_ptr<iir::StencilFunCallExpr>&
  /// expr) in StencilInstantiation.cpp). An AccessID of 0 indicates we are not in this scenario.
  int AccessIDOfCaller_;

  /// Nesting of the scopes
  int scopeDepth_;

  /// New expression which will be substitued for the `StencilFunCallExpr` (may be NULL)
  std::shared_ptr<iir::Expr> newExpr_;

  /// Scope of the current argument list being parsed
  struct ArgListScope {
    ArgListScope(const std::shared_ptr<iir::StencilFunctionInstantiation>& function)
        : Function(function), ArgumentIndex(0) {}

    const std::shared_ptr<iir::StencilFunctionInstantiation>& Function;
    int ArgumentIndex;
  };

  std::stack<ArgListScope> argListScope_;

public:
  Inliner(PassInlining::InlineStrategy strategy,
          const std::shared_ptr<iir::StencilFunctionInstantiation>& stencilFunctioninstantiation,
          const std::shared_ptr<iir::Stmt>& oldStmt,
          std::vector<std::shared_ptr<iir::Stmt>>& newStmts, int AccessIDOfCaller,
          const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation)
      : strategy_(strategy), curStencilFunctioninstantiation_(stencilFunctioninstantiation),
        instantiation_(stencilInstantiation), metadata_(stencilInstantiation->getMetaData()),
        oldStmt_(oldStmt), newStmts_(newStmts), AccessIDOfCaller_(AccessIDOfCaller), scopeDepth_(0),
        newExpr_(nullptr) {}

  /// @brief Get the new expression which will be substitued for the `StencilFunCallExpr` of this
  /// `StencilFunctionInstantiation` (may be NULL)
  std::shared_ptr<iir::Expr> getNewExpr() const { return newExpr_; }

  virtual void visit(const std::shared_ptr<iir::BlockStmt>& stmt) override {
    scopeDepth_++;
    for(const auto& s : stmt->getStatements())
      s->accept(*this);
    scopeDepth_--;
  }

  virtual void visit(const std::shared_ptr<iir::LoopStmt>& stmt) override {
    stmt->getBlockStmt()->accept(*this);
  }

  void appendNewStatement(const std::shared_ptr<iir::Stmt>& stmt) {
    stmt->getData<iir::IIRStmtData>().StackTrace = oldStmt_->getData<iir::IIRStmtData>().StackTrace;
    if(scopeDepth_ == 1) {
      newStmts_.emplace_back(stmt);
    }
  }

  virtual void visit(const std::shared_ptr<iir::ExprStmt>& stmt) override {
    appendNewStatement(stmt);
    stmt->getExpr()->accept(*this);
  }

  virtual void visit(const std::shared_ptr<iir::ReturnStmt>& stmt) override {
    DAWN_ASSERT_MSG(scopeDepth_ == 1, "cannot inline nested return statement!");

    // Instead of returning a value, we assign it to a local variable
    if(AccessIDOfCaller_ == 0) {
      // We are *not* called within an arugment list of a stencil function, meaning we can store the
      // return value in a local variable.

      // Declare and register the variable
      const bool keepVarName = false; // We want the full name (completed with access ID)
      auto newStmt = metadata_.declareVar(keepVarName, curStencilFunctioninstantiation_->getName(),
                                          dawn::Type(BuiltinTypeID::Float, CVQualifier::Const),
                                          stmt->getExpr());
      // Add it to the AST
      appendNewStatement(newStmt);

      // Set the access ID to the access expression
      auto varAccessExpr = std::make_shared<iir::VarAccessExpr>(newStmt->getName());
      varAccessExpr->getData<iir::IIRAccessExprData>().AccessID =
          std::make_optional(iir::getAccessID(newStmt));

      newExpr_ = varAccessExpr;

    } else {
      // We are called within an arugment list of a stencil function, we thus need to store the
      // return value in temporary storage (we only land here if we do precomputations).
      auto returnFieldName = iir::InstantiationHelper::makeTemporaryFieldname(
          curStencilFunctioninstantiation_->getName(), AccessIDOfCaller_);

      newExpr_ = std::make_shared<iir::FieldAccessExpr>(returnFieldName);
      auto newStmt =
          iir::makeExprStmt(std::make_shared<iir::AssignmentExpr>(newExpr_, stmt->getExpr()));
      appendNewStatement(newStmt);

      // Promote the "temporary" storage we used to mock the argument to an actual temporary field

      // First figure out the dimensions
      // TODO sparse_dim: Should be supported: should use same code used for checks on correct
      // dimensionality in statements.
      if(instantiation_->getIIR()->getGridType() != ast::GridType::Cartesian)
        dawn_unreachable(
            "Currently promotion to temporary field is not supported for unstructured grids.");
      sir::FieldDimensions fieldDims{sir::HorizontalFieldDimension(ast::cartesian, {true, true}),
                                     true};
      // Register the temporary in the metadata
      metadata_.insertAccessOfType(iir::FieldAccessType::StencilTemporary, AccessIDOfCaller_,
                                   returnFieldName);
      metadata_.setFieldDimensions(AccessIDOfCaller_, std::move(fieldDims));

      // Update the access expression with the access id of the field
      std::dynamic_pointer_cast<iir::FieldAccessExpr>(newExpr_)
          ->getData<iir::IIRAccessExprData>()
          .AccessID = std::make_optional(AccessIDOfCaller_);
    }

    // Resolve the actual expression of the return statement
    stmt->getExpr()->accept(*this);
  }

  void visit(const std::shared_ptr<iir::IfStmt>& stmt) override {
    appendNewStatement(stmt);
    stmt->getCondExpr()->accept(*this);

    stmt->getThenStmt()->accept(*this);
    if(stmt->hasElse())
      stmt->getElseStmt()->accept(*this);
  }

  void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override {
    int AccessID = iir::getAccessID(stmt);
    const std::string& name = curStencilFunctioninstantiation_->getFieldNameFromAccessID(AccessID);
    metadata_.addAccessIDNamePair(AccessID, name);
    metadata_.addAccessIDToLocalVariableDataPair(AccessID, iir::LocalVariableData{});

    // Push back the statement and move on
    appendNewStatement(stmt);

    // Resolve the RHS
    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);
  }

  void visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>&) override {}
  void visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) override {
    expr->getInit()->accept(*this);
    expr->getRhs()->accept(*this);
  }
  void visit(const std::shared_ptr<iir::StencilCallDeclStmt>&) override {}
  void visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>&) override {}

  void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<iir::UnaryOperator>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<iir::BinaryOperator>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<iir::TernaryOperator>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  virtual void visit(const std::shared_ptr<iir::FunCallExpr>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override {
    // This is a nested stencil function call (i.e a stencil function call within the current
    // stencil function)
    std::shared_ptr<iir::StencilFunctionInstantiation> func =
        curStencilFunctioninstantiation_->getStencilFunctionInstantiation(expr);
    metadata_.insertExprToStencilFunctionInstantiation(func);

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

    int oldSize = newStmts_.size();
    // Try to inline the stencil-function
    auto inlineResult = tryInlineStencilFunction(strategy_, func, oldStmt_, newStmts_,
                                                 AccessIDOfCaller, instantiation_);

    // Computes the iterator to the statement of our current stencil-function call
    const auto funCallStmtIt = newStmts_.begin() + (oldSize - 1);
    if(inlineResult.first) {
      if(func->hasReturn()) {
        std::shared_ptr<Inliner>& inliner = inlineResult.second;
        DAWN_ASSERT(inliner);
        DAWN_ASSERT(inliner->getNewExpr());

        // We need to change the current statement s.t instead of calling the stencil-function it
        // accesses the precomputed value. In addition, this statement needs to be last again (we
        // push backed all the new statements).
        iir::replaceOldExprWithNewExprInStmt(*funCallStmtIt, expr, inliner->getNewExpr());
        // Moves it to the end
        std::rotate(funCallStmtIt, std::next(funCallStmtIt), newStmts_.end());
      } else
        // Erase the statement of the original stencil function call.
        newStmts_.erase(funCallStmtIt);

      // Remove the function
      metadata_.removeStencilFunctionInstantiation(expr, curStencilFunctioninstantiation_);

    } else {
      // Inlining failed, transfer ownership
      metadata_.insertExprToStencilFunctionInstantiation(func);
    }

    if(!argListScope_.empty())
      argListScope_.top().ArgumentIndex++;
  }

  void visit(const std::shared_ptr<iir::StencilFunArgExpr>&) override {
    if(!argListScope_.empty())
      argListScope_.top().ArgumentIndex++;
  }

  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override {

    std::string callerName = metadata_.getFieldNameFromAccessID(iir::getAccessID(expr));
    expr->setName(callerName);

    if(expr->isArrayAccess())
      expr->getIndex()->accept(*this);
  }

  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {

    std::string callerName = metadata_.getFieldNameFromAccessID(iir::getAccessID(expr));
    expr->setName(callerName);

    // Set the fully evaluated offset as the new offset of the field. Note that this renders the
    // AST of the current stencil function incorrent which is why it needs to be removed!
    expr->setPureOffset(curStencilFunctioninstantiation_->evalOffsetOfFieldAccessExpr(expr, true));

    if(!argListScope_.empty())
      argListScope_.top().ArgumentIndex++;
  }

  void visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) override {
    int AccessID = iir::getAccessID(expr);
    metadata_.insertAccessOfType(iir::FieldAccessType::Literal, AccessID, expr->getValue());
  }
};

/// @brief Detect inline candidates
class DetectInlineCandiates : public iir::ASTVisitorForwarding {
  PassInlining::InlineStrategy strategy_;
  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;

  /// The statement we are currently analyzing
  iir::BlockStmt::StmtConstIterator oldStmt_;

  /// If non-empty the `oldStmt` will be appended to `newStmts` with the given replacements
  std::unordered_map<std::shared_ptr<iir::Expr>, std::shared_ptr<iir::Expr>>
      replacmentOfOldStmtMap_;

  /// The new list of Stmt which can serve as a replacement for `oldStmt`
  std::vector<std::shared_ptr<iir::Stmt>> newStmts_;

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
  using Base = iir::ASTVisitorForwarding;

  DetectInlineCandiates(PassInlining::InlineStrategy strategy,
                        const std::shared_ptr<iir::StencilInstantiation>& instantiation)
      : strategy_(strategy), instantiation_(instantiation), inlineCandiatesFound_(false) {}

  /// @brief Process the given statement
  void processStatement(iir::BlockStmt::StmtConstIterator stmt) {
    // Reset the state
    inlineCandiatesFound_ = false;
    oldStmt_ = stmt;
    newStmts_.clear();

    // Detect the stencil functions suitable for inlining
    (*oldStmt_)->accept(*this);
  }

  /// @brief Atleast one inline candiate was found and the given `stmt` should be replaced with
  /// `getNewStatements`
  bool inlineCandiatesFound() const { return inlineCandiatesFound_; }

  /// @brief Get the newly computed statements which can be substituted for the given `stmt`
  ///
  /// Note that the accesses are not computed!
  std::vector<std::shared_ptr<iir::Stmt>>& getNewStatements() {
    if(!replacmentOfOldStmtMap_.empty()) {
      newStmts_.push_back(*std::make_move_iterator(oldStmt_));

      for(const auto& oldNewPair : replacmentOfOldStmtMap_)
        iir::replaceOldExprWithNewExprInStmt(newStmts_[newStmts_.size() - 1], oldNewPair.first,
                                             oldNewPair.second);

      // Clear the map in case someone would call getNewStatments multiple times
      replacmentOfOldStmtMap_.clear();
    }

    return newStmts_;
  }

  void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override {
    std::shared_ptr<iir::StencilFunctionInstantiation> func =
        instantiation_->getMetaData().getStencilFunctionInstantiation(expr);

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

    auto inlineResult = tryInlineStencilFunction(strategy_, func, *oldStmt_, newStmts_,
                                                 AccessIDOfCaller, instantiation_);

    inlineCandiatesFound_ |= inlineResult.first;
    if(inlineResult.first) {

      // Replace `StencilFunCallExpr` with an access to a storage or variable containing the result
      // of the stencil function call
      if(inlineResult.second->getNewExpr())
        replacmentOfOldStmtMap_.emplace(expr, inlineResult.second->getNewExpr());

      // Remove the stencil-function (`nullptr` means we don't have a nested stencil function)
      instantiation_->getMetaData().removeStencilFunctionInstantiation(expr, nullptr);
    }

    if(!argListScope_.empty())
      argListScope_.top().ArgumentIndex++;
  }

  void visit(const std::shared_ptr<iir::StencilFunArgExpr>&) override {
    if(!argListScope_.empty())
      argListScope_.top().ArgumentIndex++;
  }

  void visit(const std::shared_ptr<iir::FieldAccessExpr>&) override {
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
static std::pair<bool, std::shared_ptr<Inliner>>
tryInlineStencilFunction(PassInlining::InlineStrategy strategy,
                         const std::shared_ptr<iir::StencilFunctionInstantiation>& stencilFunc,
                         const std::shared_ptr<iir::Stmt>& oldStmt,
                         std::vector<std::shared_ptr<iir::Stmt>>& newStmts, int AccessIDOfCaller,
                         const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  // Function which do not return a value are *always* inlined. Function which do return a value
  // are only inlined if we favor precomputations.
  if(!stencilFunc->hasReturn() || strategy == PassInlining::InlineStrategy::ComputationsOnTheFly) {
    auto inliner = std::make_shared<Inliner>(strategy, stencilFunc, oldStmt, newStmts,
                                             AccessIDOfCaller, stencilInstantiation);
    stencilFunc->getAST()->accept(*inliner);
    return std::pair<bool, std::shared_ptr<Inliner>>(true, std::move(inliner));
  }
  return std::pair<bool, std::shared_ptr<Inliner>>(false, nullptr);
}

} // anonymous namespace

PassInlining::PassInlining(OptimizerContext& context, InlineStrategy strategy)
    : Pass(context, "PassInlining", true), strategy_(strategy) {}

bool PassInlining::run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  DetectInlineCandiates inliner(strategy_, stencilInstantiation);

  // Iterate all statements (top -> bottom)
  for(const auto& stagePtr : iterateIIROver<iir::Stage>(*(stencilInstantiation->getIIR()))) {
    iir::Stage& stage = *stagePtr;
    for(auto& doMethod : stage.getChildren()) {
      for(auto stmtIt = doMethod->getAST().getStatements().begin();
          stmtIt != doMethod->getAST().getStatements().end(); ++stmtIt) {
        inliner.processStatement(stmtIt);

        if(inliner.inlineCandiatesFound()) {
          auto& newStmtList = inliner.getNewStatements();
          // Compute the accesses of the new statements
          computeAccesses(stencilInstantiation->getMetaData(), newStmtList);
          // Erase the old stmt ...
          stmtIt = doMethod->getAST().erase(stmtIt);

          // ... and insert the new ones
          // newStmtList will be cleared at the next for iteration, so it is safe to move the
          // elements here
          stmtIt = doMethod->getAST().insert(stmtIt, std::make_move_iterator(newStmtList.begin()),
                                             std::make_move_iterator(newStmtList.end()));

          std::advance(stmtIt, newStmtList.size() - 1);
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
