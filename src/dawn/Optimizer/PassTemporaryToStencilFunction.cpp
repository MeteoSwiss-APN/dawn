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
sir::Interval intervalToSIRInterval(Interval interval) {
  return sir::Interval(interval.lowerLevel(), interval.upperLevel(), interval.lowerOffset(),
                       interval.upperOffset());
}

/// @brief some properties of the temporary being replaced
struct TemporaryFunctionProperties {
  std::shared_ptr<StencilFunCallExpr>
      stencilFunCallExpr_;        ///< stencil function call that will replace the tmp
  std::vector<int> accessIDArgs_; ///< access IDs of the args that are needed to compute the tmp
  std::shared_ptr<sir::StencilFunction>
      sirStencilFunction_; ///< sir stencil function of the tmp being created
  std::shared_ptr<FieldAccessExpr>
      tmpFieldAccessExpr_; ///< FieldAccessExpr of the tmp captured for replacement
};

/// @brief visitor that will detect assignment (i.e. computations) to a temporary,
/// it will create a sir::StencilFunction out of this computation, and replace the assignment
/// expression in the AST by a NOExpr.
class TmpAssignment : public ASTVisitorPostOrder, public NonCopyable {
protected:
  const std::shared_ptr<StencilInstantiation>& instantiation_;
  sir::Interval interval_;
  std::shared_ptr<sir::StencilFunction> tmpFunction_;
  std::vector<int> accessIDs_;

  int accessID_ = -1;
  std::shared_ptr<FieldAccessExpr> tmpFieldAccessExpr_ = nullptr;

public:
  TmpAssignment(const std::shared_ptr<StencilInstantiation>& instantiation,
                sir::Interval const& interval)
      : instantiation_(instantiation), interval_(interval), tmpFunction_(nullptr) {}

  virtual ~TmpAssignment() {}

  int temporaryFieldAccessID() const { return accessID_; }

  const std::vector<int>& getAccessIDs() const { return accessIDs_; }

  std::shared_ptr<FieldAccessExpr> getTemporaryFieldAccessExpr() { return tmpFieldAccessExpr_; }

  std::shared_ptr<sir::StencilFunction> temporaryStencilFunction() { return tmpFunction_; }

  bool foundTemporaryToReplace() { return (tmpFunction_ != nullptr); }

  /// @brief pre visit the node. The assignment expr visit will only continue processing the visitor
  /// for the right hand side of the =. Therefore all accesses capture here correspond to arguments
  /// for computing the tmp. They are captured as arguments of the stencil function being created
  virtual bool preVisitNode(std::shared_ptr<FieldAccessExpr> const& expr) override {
    DAWN_ASSERT(tmpFunction_);
    for(int idx : expr->getArgumentMap()) {
      DAWN_ASSERT(idx == -1);
    }
    for(int off : expr->getArgumentOffset())
      DAWN_ASSERT(off == 0);

    if(!tmpFunction_->hasArg(expr->getName()) && expr != tmpFieldAccessExpr_) {

      int genLineKey = static_cast<std::underlying_type<SourceLocation::ReservedSL>::type>(
          SourceLocation::ReservedSL::SL_Generated);
      tmpFunction_->Args.push_back(
          std::make_shared<sir::Field>(expr->getName(), SourceLocation(genLineKey, genLineKey)));

      accessIDs_.push_back(instantiation_->getAccessIDFromExpr(expr));
    }
    // continue traversing
    return true;
  }

  /// @brief capture a tmp computation
  virtual bool preVisitNode(std::shared_ptr<AssignmentExpr> const& expr) override {
    if(isa<FieldAccessExpr>(*(expr->getLeft()))) {

      // return and stop traversing the AST if the left hand of the =  is not a temporary
      int accessID = instantiation_->getAccessIDFromExpr(expr->getLeft());
      if(!instantiation_->isTemporaryField(accessID))
        return false;
      tmpFieldAccessExpr_ = std::dynamic_pointer_cast<FieldAccessExpr>(expr->getLeft());
      accessID_ = accessID;

      // otherwise we create a new stencil function
      std::string tmpFieldName = instantiation_->getNameFromAccessID(accessID_);
      tmpFunction_ = std::make_shared<sir::StencilFunction>();

      tmpFunction_->Name = tmpFieldName + "_OnTheFly";
      tmpFunction_->Loc = expr->getSourceLocation();
      tmpFunction_->Intervals.push_back(std::make_shared<sir::Interval>(interval_));

      return true;
    }
    return false;
  }
  ///@brief once the "tmp= fn(args)" has been captured, the new stencil function to compute the
  /// temporary is finalized and the assignment is replaced by a NOExpr
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
};

/// @brief visitor that will capture all read accesses to the temporary. The offset used to access
/// the temporary is extracted and pass to all the stencil function arguments. A new stencil
/// function instantiation is then created and the field access expression replaced by a stencil
/// function call
class TmpReplacement : public ASTVisitorPostOrder, public NonCopyable {
protected:
  const std::shared_ptr<StencilInstantiation>& instantiation_;
  std::unordered_map<int, TemporaryFunctionProperties> const& temporaryFieldAccessIDToFunctionCall_;
  const sir::Interval interval_;
  std::shared_ptr<std::vector<sir::StencilCall*>> stackTrace_;

  unsigned int numTmpReplaced_ = 0;

public:
  TmpReplacement(const std::shared_ptr<StencilInstantiation>& instantiation,
                 std::unordered_map<int, TemporaryFunctionProperties> const&
                     temporaryFieldAccessIDToFunctionCall,
                 const sir::Interval sirInterval,
                 std::shared_ptr<std::vector<sir::StencilCall*>> stackTrace)
      : instantiation_(instantiation),
        temporaryFieldAccessIDToFunctionCall_(temporaryFieldAccessIDToFunctionCall),
        interval_(sirInterval), stackTrace_(stackTrace) {}

  virtual ~TmpReplacement() {}

  static std::string offsetToString(int a) {
    return ((a < 0) ? "minus" : "") + std::to_string(std::abs(a));
  }

  /// @brief create the name of a newly created stencil function associated to a tmp computations
  std::string makeOnTheFlyFunctionName(const std::shared_ptr<FieldAccessExpr>& expr) {
    return expr->getName() + "_OnTheFly" + "_i" + offsetToString(expr->getOffset()[0]) + "_j" +
           offsetToString(expr->getOffset()[1]) + "_k" + offsetToString(expr->getOffset()[2]);
  }

  unsigned int getNumTmpReplaced() { return numTmpReplaced_; }
  void resetNumTmpReplaced() { numTmpReplaced_ = 0; }

  /// @brief previsit the access to a temporary. Finalize the stencil function instantiation and
  /// recompute its <statement,accesses> pairs
  virtual bool preVisitNode(std::shared_ptr<FieldAccessExpr> const& expr) override {
    int accessID = instantiation_->getAccessIDFromExpr(expr);
    if(!temporaryFieldAccessIDToFunctionCall_.count(accessID))
      return false;
    // TODO we need to version to tmp function generation, in case tmp is recomputed multiple times
    std::string callee = expr->getName() + "_OnTheFly";
    std::shared_ptr<StencilFunctionInstantiation> stencilFun =
        instantiation_->getStencilFunctionInstantiationCandidate(callee);

    std::string fnClone = makeOnTheFlyFunctionName(expr);

    // the temporary was not replaced by a stencil function by previous visitor, for some reasons it
    // was not supported
    if(instantiation_->hasStencilFunctionInstantiation(fnClone))
      return true;

    DAWN_ASSERT(temporaryFieldAccessIDToFunctionCall_.count(accessID));

    // retrieve the sir stencil function definition
    std::shared_ptr<sir::StencilFunction> sirStencilFunction =
        temporaryFieldAccessIDToFunctionCall_.at(accessID).sirStencilFunction_;

    // we create a new sir stencil function, since its name is demangled from the offsets.
    // for example, for a tmp(i+1) the stencil function is named as tmp_OnTheFly_iplus1
    std::shared_ptr<sir::StencilFunction> sirStencilFunctionInstance =
        std::make_shared<sir::StencilFunction>(*sirStencilFunction);

    sirStencilFunctionInstance->Name = fnClone;

    // TODO is this really needed, we only change the name, can we map multiple function
    // instantiations (i.e. different offsets) to the same sir stencil function
    // insert the sir::stencilFunction into the StencilInstantiation
    instantiation_->insertStencilFunctionIntoSIR(sirStencilFunctionInstance);

    std::shared_ptr<StencilFunctionInstantiation> cloneStencilFun =
        instantiation_->cloneStencilFunctionCandidate(stencilFun, fnClone);

    auto& accessIDsOfArgs = temporaryFieldAccessIDToFunctionCall_.at(accessID).accessIDArgs_;

    // here we create the arguments of the stencil function instantiation.
    // find the accessID of all args, and create a new FieldAccessExpr with an offset corresponding
    // to the offset used to access the temporary
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

    std::unordered_map<std::string, int> fieldsMap;

    const auto& arguments = cloneStencilFun->getArguments();
    for(std::size_t argIdx = 0; argIdx < arguments.size(); ++argIdx) {
      if(sir::Field* field = dyn_cast<sir::Field>(arguments[argIdx].get())) {
        int AccessID = cloneStencilFun->getCallerAccessIDOfArgField(argIdx);
        fieldsMap[field->Name] = AccessID;
      }
    }

    // recompute the list of <statement, accesses> pairs
    StatementMapper statementMapper(instantiation_.get(), stackTrace_,
                                    cloneStencilFun->getStatementAccessesPairs(), interval_,
                                    fieldsMap, cloneStencilFun);

    cloneStencilFun->getAST()->accept(statementMapper);

    // final checks
    cloneStencilFun->checkFunctionBindings();

    return true;
  }

  /// @brief replace the access to a temporary by a stencil function call expression
  virtual std::shared_ptr<Expr>
  postVisitNode(std::shared_ptr<FieldAccessExpr> const& expr) override {

    int accessID = instantiation_->getAccessIDFromExpr(expr);
    // if the field access is not identified as a temporary being replaced just return without
    // substitution
    if(!temporaryFieldAccessIDToFunctionCall_.count(accessID))
      return expr;

    // TODO we need to version to tmp function generation, in case tmp is recomputed multiple times
    std::string callee = makeOnTheFlyFunctionName(expr);

    auto stencilFunCall = instantiation_->getStencilFunctionInstantiation(callee)->getExpression();

    numTmpReplaced_++;
    return stencilFunCall;
  }
};

} // anonymous namespace

PassTemporaryToStencilFunction::PassTemporaryToStencilFunction()
    : Pass("PassTemporaryToStencilFunction") {}

bool PassTemporaryToStencilFunction::run(
    const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {

  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  if(!context->getOptions().PassTmpToFunction)
    return true;

  DAWN_ASSERT(context);

  for(auto& stencilPtr : stencilInstantiation->getStencils()) {

    // Iterate multi-stages backwards
    for(auto multiStage : stencilPtr->getMultiStages()) {
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

              // find patterns like tmp = fn(args)...;
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

                // all the temporary computations captured are stored in this map of <ID, tmp
                // properties>
                // for later use of the replacer visitor
                temporaryFieldExprToFunction.emplace(
                    stencilInstantiation->getAccessIDFromExpr(
                        tmpAssignment.getTemporaryFieldAccessExpr()),
                    TemporaryFunctionProperties{stencilFunCallExpr, tmpAssignment.getAccessIDs(),
                                                stencilFunction,
                                                tmpAssignment.getTemporaryFieldAccessExpr()});

                // first instantiation of the stencil function that is inserted in the IIR as a
                // candidate stencil function
                auto stencilFun = stencilInstantiation->makeStencilFunctionInstantiation(
                    stencilFunCallExpr, stencilFunction, ast, sirInterval, nullptr);

                int argID = 0;
                for(auto accessID : tmpAssignment.getAccessIDs()) {
                  stencilFun->setCallerAccessIDOfArgField(argID++, accessID);
                }
              }

              // run the replacer visitor
              TmpReplacement tmpReplacement(stencilInstantiation, temporaryFieldExprToFunction,
                                            sirInterval, stmt->StackTrace);
              stmt->ASTStmt->acceptAndReplace(tmpReplacement);
              // flag if a least a tmp has been replaced within this stage
              isATmpReplaced = isATmpReplaced || (tmpReplacement.getNumTmpReplaced() != 0);

              if(tmpReplacement.getNumTmpReplaced() != 0) {
                std::vector<std::shared_ptr<StatementAccessesPair>> listStmtPair;

                // TODO need to combine call to StatementMapper with computeAccesses in a clean way
                // since the AST has changed, we need to recompute the <statement,accesses> pairs
                // of the stage

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

      std::cout << "\nPASS: " << getName() << "; stencil: " << stencilInstantiation->getName();

      if(temporaryFieldExprToFunction.empty())
        std::cout << "no replacement found";

      for(auto tmpFieldPair : temporaryFieldExprToFunction) {
        int accessID = tmpFieldPair.first;
        auto tmpProperties = tmpFieldPair.second;
        if(context->getOptions().ReportPassTmpToFunction)

          std::cout << " [ replace tmp:" << stencilInstantiation->getNameFromAccessID(accessID)
                    << "; line : " << tmpProperties.tmpFieldAccessExpr_->getSourceLocation().Line
                    << " ] ";
      }
      std::cout << std::endl;
    }
  }

  return true;
}

} // namespace dawn
