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

#include "dawn/Optimizer/PassTemporaryToStencilFunction.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/Optimizer/Stencil.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"

namespace dawn {

namespace {
// TODO just have one interval class, we dont need two
sir::Interval intervalToSIRInterval(Interval interval) {
  return sir::Interval(interval.lowerLevel(), interval.upperLevel(), interval.lowerOffset(),
                       interval.upperOffset());
}

struct TemporaryFunctionProperties {
  std::shared_ptr<StencilFunCallExpr> stencilFunCallExpr_;
  std::vector<int> accessIDArgs_;
  std::shared_ptr<sir::StencilFunction> sirStencilFunction_;
};

class TmpAssignment : public ASTVisitorPostOrder, public NonCopyable {
protected:
  std::shared_ptr<StencilInstantiation> instantiation_;
  sir::Interval interval_;
  std::shared_ptr<sir::StencilFunction> tmpFunction_;
  std::vector<int> accessIDs_;

  // TODO remove, not used
  std::shared_ptr<std::vector<std::shared_ptr<FieldAccessExpr>>> tmpComputationArgs_;
  //  std::unordered_set<std::string> insertedFields_;
  int accessID_ = -1;
  std::shared_ptr<FieldAccessExpr> tmpFieldAccessExpr_ = nullptr;

public:
  TmpAssignment(std::shared_ptr<StencilInstantiation> instantiation, sir::Interval const& interval)
      : instantiation_(instantiation), interval_(interval), tmpComputationArgs_(nullptr) {}

  virtual ~TmpAssignment() {}

  int temporaryFieldAccessID() const { return accessID_; }

  std::shared_ptr<std::vector<std::shared_ptr<FieldAccessExpr>>> temporaryComputationArgs() {
    return tmpComputationArgs_;
  }

  std::vector<int> const& getAccessIDs() { return accessIDs_; }

  std::shared_ptr<FieldAccessExpr> getTemporaryFieldAccessExpr() { return tmpFieldAccessExpr_; }

  std::shared_ptr<sir::StencilFunction> temporaryStencilFunction() { return tmpFunction_; }

  bool foundTemporaryToReplace() { return (tmpFunction_ != nullptr); }

  /// @name Expression implementation
  /// @{
  virtual bool preVisitNode(std::shared_ptr<FieldAccessExpr> const& expr) override {
    DAWN_ASSERT(tmpFunction_);
    for(int idx : expr->getArgumentMap()) {
      DAWN_ASSERT(idx == -1);
    }
    for(int off : expr->getArgumentOffset())
      DAWN_ASSERT(off == 0);

    if(tmpComputationArgs_ == nullptr)
      tmpComputationArgs_ = std::make_shared<std::vector<std::shared_ptr<FieldAccessExpr>>>();

    tmpComputationArgs_->push_back(expr);
    if(!tmpFunction_->hasArg(expr->getName()) && expr != tmpFieldAccessExpr_) {
      tmpFunction_->Args.push_back(std::make_shared<sir::Field>(expr->getName(), SourceLocation{}));
      accessIDs_.push_back(instantiation_->getAccessIDFromExpr(expr));
    }
    return true;
  }

  /// @name Expression implementation
  /// @{

  virtual bool preVisitNode(std::shared_ptr<AssignmentExpr> const& expr) override {
    if(isa<FieldAccessExpr>(*(expr->getLeft()))) {
      tmpFieldAccessExpr_ = std::dynamic_pointer_cast<FieldAccessExpr>(expr->getLeft());
      accessID_ = instantiation_->getAccessIDFromExpr(expr->getLeft());

      if(!instantiation_->isTemporaryField(accessID_))
        return false;

      std::string tmpFieldName = instantiation_->getNameFromAccessID(accessID_);

      tmpFunction_ = std::make_shared<sir::StencilFunction>();

      tmpFunction_->Name = tmpFieldName + "_OnTheFly";
      tmpFunction_->Loc = expr->getSourceLocation();
      // TODO cretae a interval->sir::interval converter
      tmpFunction_->Intervals.push_back(std::make_shared<sir::Interval>(interval_));

      return true;
    }
    return false;
  }
  virtual std::shared_ptr<Expr>
  postVisitNode(std::shared_ptr<AssignmentExpr> const& expr) override {
    if(isa<FieldAccessExpr>(*(expr->getLeft()))) {
      accessID_ = instantiation_->getAccessIDFromExpr(expr->getLeft());
      if(!instantiation_->isTemporaryField(accessID_))
        return expr;

      DAWN_ASSERT(tmpFunction_);

      auto functionExpr = expr->getRight()->clone();

      auto retStmt = std::make_shared<ReturnStmt>(functionExpr);

      std::shared_ptr<BlockStmt> root = std::make_shared<BlockStmt>();
      root->push_back(retStmt);
      std::shared_ptr<AST> ast = std::make_shared<AST>(root);
      tmpFunction_->Asts.push_back(ast);

      return std::make_shared<NOPExpr>();
    }
    return expr;
  }
  /// @}
};

class TmpReplacement : public ASTVisitorPostOrder, public NonCopyable {
protected:
  std::shared_ptr<StencilInstantiation> instantiation_;
  std::unordered_map<int, TemporaryFunctionProperties> const& temporaryFieldAccessIDToFunctionCall_;
  std::shared_ptr<std::vector<std::shared_ptr<FieldAccessExpr>>> tmpComputationArgs_;
  const sir::Interval interval_;
  std::shared_ptr<std::vector<sir::StencilCall*>> stackTrace_;

  unsigned int numTmpReplaced_ = 0;

public:
  TmpReplacement(std::shared_ptr<StencilInstantiation> instantiation,
                 std::unordered_map<int, TemporaryFunctionProperties> const&
                     temporaryFieldAccessIDToFunctionCall,
                 const sir::Interval sirInterval,
                 std::shared_ptr<std::vector<sir::StencilCall*>> stackTrace)
      : instantiation_(instantiation),
        temporaryFieldAccessIDToFunctionCall_(temporaryFieldAccessIDToFunctionCall),
        interval_(sirInterval), stackTrace_(stackTrace) {}

  virtual ~TmpReplacement() {}

  std::string makeOnTheFlyFunctionName(std::shared_ptr<FieldAccessExpr> expr) {
    return expr->getName() + "_OnTheFly" + "_i" + std::to_string(expr->getOffset()[0]) + "_j" +
           std::to_string(expr->getOffset()[1]) + "_k" + std::to_string(expr->getOffset()[2]);
  }

  unsigned int getNumTmpReplaced() { return numTmpReplaced_; }
  void resetNumTmpReplaced() { numTmpReplaced_ = 0; }

  /// @name Expression implementation
  /// @{
  virtual bool preVisitNode(std::shared_ptr<FieldAccessExpr> const& expr) override {
    int accessID = instantiation_->getAccessIDFromExpr(expr);
    if(!temporaryFieldAccessIDToFunctionCall_.count(accessID))
      return false;
    // TODO we need to version to tmp function generation, in case tmp is recomputed multiple times
    std::string callee = expr->getName() + "_OnTheFly";
    std::shared_ptr<StencilFunctionInstantiation> stencilFun =
        instantiation_->getStencilFunctionInstantiationCandidate(callee);

    std::string fnClone = makeOnTheFlyFunctionName(expr);

    if(instantiation_->hasStencilFunctionInstantiation(fnClone))
      return true;

    DAWN_ASSERT(temporaryFieldAccessIDToFunctionCall_.count(accessID));

    std::shared_ptr<sir::StencilFunction> sirStencilFunction =
        temporaryFieldAccessIDToFunctionCall_.at(accessID).sirStencilFunction_;

    std::shared_ptr<sir::StencilFunction> sirStencilFunctionInstance =
        std::make_shared<sir::StencilFunction>(*sirStencilFunction);

    sirStencilFunctionInstance->Name = fnClone;

    // insert the sir::stencilFunction into the StencilInstantiation
    instantiation_->insertStencilFunctionIntoSIR(sirStencilFunctionInstance);

    std::shared_ptr<StencilFunctionInstantiation> cloneStencilFun =
        instantiation_->cloneStencilFunctionCandidate(stencilFun, fnClone);

    auto& accessIDsOfArgs = temporaryFieldAccessIDToFunctionCall_.at(accessID).accessIDArgs_;

    for(auto accessID_ : (accessIDsOfArgs)) {
      std::shared_ptr<FieldAccessExpr> arg = std::make_shared<FieldAccessExpr>(
          instantiation_->getNameFromAccessID(accessID_), expr->getOffset());
      cloneStencilFun->getExpression()->insertArgument(arg);

      instantiation_->mapExprToAccessID(arg, accessID_);
    }

    // TODO coming from stencil functions is not yet supported
    for(int idx : expr->getArgumentMap()) {
      DAWN_ASSERT(idx == -1);
    }
    for(int off : expr->getArgumentOffset())
      DAWN_ASSERT(off == 0);

    const auto& argToAccessIDMap = stencilFun->ArgumentIndexToCallerAccessIDMap();
    for(auto pair : argToAccessIDMap) {
      int accessID_ = pair.second;
      cloneStencilFun->setCallerInitialOffsetFromAccessID(accessID_, expr->getOffset());
    }

    instantiation_->finalizeStencilFunctionSetup(cloneStencilFun);

    StatementMapper statementMapper(instantiation_.get(), stackTrace_,
                                    cloneStencilFun->getStatementAccessesPairs(), interval_,
                                    instantiation_->getNameToAccessIDMap(), cloneStencilFun);

    cloneStencilFun->getAST()->accept(statementMapper);

    cloneStencilFun->checkFunctionBindings();

    return true;
  }

  virtual std::shared_ptr<Expr>
  postVisitNode(std::shared_ptr<FieldAccessExpr> const& expr) override {

    int accessID = instantiation_->getAccessIDFromExpr(expr);
    if(!temporaryFieldAccessIDToFunctionCall_.count(accessID))
      return expr;

    // TODO we need to version to tmp function generation, in case tmp is recomputed multiple times
    std::string callee = makeOnTheFlyFunctionName(expr);

    auto stencilFunCall = instantiation_->getStencilFunctionInstantiation(callee)->getExpression();

    numTmpReplaced_++;
    return stencilFunCall;
  }

  /// @}
};

} // anonymous namespace

PassTemporaryToStencilFunction::PassTemporaryToStencilFunction()
    : Pass("PassTemporaryToStencilFunction") {}

bool PassTemporaryToStencilFunction::run(
    std::shared_ptr<StencilInstantiation> stencilInstantiation) {
  for(auto& stencilPtr : stencilInstantiation->getStencils()) {
    Stencil& stencil = *stencilPtr;

    // Iterate multi-stages backwards
    for(auto multiStage : stencil.getMultiStages()) {
      //      std::shared_ptr<std::vector<std::shared_ptr<FieldAccessExpr>>>
      //      temporaryComputationArgs;
      std::unordered_map<int, TemporaryFunctionProperties> temporaryFieldExprToFunction;

      for(const auto& stagePtr : multiStage->getStages()) {

        bool isATmpReplaced = false;
        for(const auto& doMethodPtr : stagePtr->getDoMethods()) {
          for(auto& stmtAccessPair : doMethodPtr->getStatementAccessesPairs()) {
            const std::shared_ptr<Statement> stmt = stmtAccessPair->getStatement();

            if(stmt->ASTStmt->getKind() != Stmt::SK_ExprStmt)
              continue;

            // TODO catch a temp expr
            {
              const Interval& interval = doMethodPtr->getInterval();
              const sir::Interval sirInterval = intervalToSIRInterval(interval);

              TmpAssignment tmpAssignment(stencilInstantiation, sirInterval);
              stmt->ASTStmt->acceptAndReplace(tmpAssignment);
              if(tmpAssignment.foundTemporaryToReplace()) {

                std::shared_ptr<sir::StencilFunction> stencilFunction =
                    tmpAssignment.temporaryStencilFunction();
                std::shared_ptr<AST> ast = stencilFunction->getASTOfInterval(sirInterval);

                DAWN_ASSERT(ast);
                DAWN_ASSERT(stencilFunction);

                std::shared_ptr<StencilFunCallExpr> stencilFunCallExpr =
                    std::make_shared<StencilFunCallExpr>(stencilFunction->Name);

                auto accessIDsOfArgs = tmpAssignment.getAccessIDs();

                temporaryFieldExprToFunction.emplace(
                    stencilInstantiation->getAccessIDFromExpr(
                        tmpAssignment.getTemporaryFieldAccessExpr()),
                    TemporaryFunctionProperties{stencilFunCallExpr, tmpAssignment.getAccessIDs(),
                                                stencilFunction});

                auto stencilFun = stencilInstantiation->makeStencilFunctionInstantiation(
                    stencilFunCallExpr, stencilFunction, ast, sirInterval, nullptr);

                int argID = 0;
                for(auto accessID : tmpAssignment.getAccessIDs()) {
                  stencilFun->setCallerAccessIDOfArgField(argID++, accessID);
                }
                //////////
              }
              TmpReplacement tmpReplacement(stencilInstantiation, temporaryFieldExprToFunction,
                                            sirInterval, stmt->StackTrace);
              stmt->ASTStmt->acceptAndReplace(tmpReplacement);
              isATmpReplaced = isATmpReplaced || (tmpReplacement.getNumTmpReplaced() != 0);

              if(tmpReplacement.getNumTmpReplaced() != 0) {
                std::vector<std::shared_ptr<StatementAccessesPair>> listStmtPair;

                // TODO need to combine call to StatementMapper with computeAccesses in a clean
                // way
                StatementMapper statementMapper(
                    stencilInstantiation.get(), stmt->StackTrace, listStmtPair, sirInterval,
                    stencilInstantiation->getNameToAccessIDMap(), nullptr);

                std::shared_ptr<BlockStmt> blockStmt =
                    std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{stmt->ASTStmt});
                blockStmt->accept(statementMapper);
                DAWN_ASSERT(listStmtPair.size() == 1);

                std::shared_ptr<StatementAccessesPair> stmtPair = listStmtPair[0];
                computeAccesses(stencilInstantiation.get(), stmtPair);

                stmtAccessPair = stmtPair;
              }
            }
          }
        }
        if(isATmpReplaced) {
          stagePtr->update();
        }
      }
    }
  }

  return true;
}

} // namespace dawn
