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
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/AccessComputation.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/Optimizer/TemporaryHandling.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/RemoveIf.hpp"

#include <sstream>

namespace dawn {

namespace {
sir::Interval intervalToSIRInterval(iir::Interval interval) {
  return sir::Interval(interval.lowerLevel(), interval.upperLevel(), interval.lowerOffset(),
                       interval.upperOffset());
}

iir::Interval sirIntervalToInterval(sir::Interval interval) {
  return iir::Interval(interval.LowerLevel, interval.UpperLevel, interval.LowerOffset,
                       interval.UpperOffset);
}

/// @brief some properties of the temporary being replaced
struct TemporaryFunctionProperties {
  std::shared_ptr<iir::StencilFunCallExpr>
      stencilFunCallExpr_;        ///< stencil function call that will replace the tmp
  std::vector<int> accessIDArgs_; ///< access IDs of the args that are needed to compute the tmp
  std::shared_ptr<sir::StencilFunction>
      sirStencilFunction_; ///< sir stencil function of the tmp being created
  std::shared_ptr<iir::FieldAccessExpr>
      tmpFieldAccessExpr_; ///< FieldAccessExpr of the tmp captured for replacement
  iir::Interval interval_; ///< interval for which the tmp definition is valid
};

///
/// @brief The LocalVariablePromotion class identifies local variables that need to be promoted to
/// temporaries because of a tmp->stencilfunction conversion
/// In the following example:
/// double a=0;
/// tmp = a*2;
/// local variable a will have to be promoted to temporary, since tmp will be evaluated on-the-fly
/// with extents
///
class LocalVariablePromotion : public iir::ASTVisitorPostOrder, public NonCopyable {
protected:
  const iir::StencilMetaInformation& metadata_;
  const iir::Stencil& stencil_;
  const std::unordered_map<int, iir::Stencil::FieldInfo>& fields_;
  const SkipIDs& skipIDs_;
  std::unordered_set<int>& localVarAccessIDs_;
  bool activate_ = false;

public:
  LocalVariablePromotion(const iir::StencilMetaInformation& metadata, const iir::Stencil& stencil,
                         const std::unordered_map<int, iir::Stencil::FieldInfo>& fields,
                         const SkipIDs& skipIDs, std::unordered_set<int>& localVarAccessIDs)
      : metadata_(metadata), stencil_(stencil), fields_(fields), skipIDs_(skipIDs),
        localVarAccessIDs_(localVarAccessIDs) {}

  virtual ~LocalVariablePromotion() override {}

  virtual bool preVisitNode(std::shared_ptr<iir::VarAccessExpr> const& expr) override {
    // TODO if inside stencil function we should get it from stencilfun

    // we process this var access only after activation, i.e. after a "tmp= ..." pattern has been
    // found. This is important to protect against var accesses in a var decl like "float var = "
    // that could occur before the visit of the assignment expression
    if(activate_) {
      localVarAccessIDs_.emplace(iir::getAccessID(expr));
    }
    return true;
  }

  /// @brief capture a tmp computation
  virtual bool preVisitNode(std::shared_ptr<iir::AssignmentExpr> const& expr) override {

    if(isa<iir::FieldAccessExpr>(*(expr->getLeft()))) {
      int accessID = iir::getAccessID(expr->getLeft());
      DAWN_ASSERT(fields_.count(accessID));
      const iir::Field& field = fields_.at(accessID).field;

      bool skip = true;
      // If at least in one ms, the id is not skipped, we will process the local var -> tmp
      // promotion
      for(const auto& ms : stencil_.getChildren()) {
        if(!ms->getFields().count(accessID)) {
          continue;
        }
        if(!skipIDs_.skipID(ms->getID(), accessID)) {
          skip = false;
          break;
        }
      }
      if(skip) {
        return false;
      }
      if(!metadata_.isAccessType(iir::FieldAccessType::StencilTemporary, accessID))
        return false;

      if(field.getExtents().isHorizontalPointwise())
        return false;

      activate_ = true;
      return true;
    }

    return false;
  }
};

/// @brief create the name of a newly created stencil function associated to a tmp computations
std::string makeOnTheFlyFunctionCandidateName(const std::shared_ptr<iir::FieldAccessExpr>& expr,
                                              const iir::Interval& interval) {
  return expr->getName() + "_OnTheFly_" + interval.toStringGen();
}

/// @brief create the name of a newly created stencil function associated to a tmp computations
std::string makeOnTheFlyFunctionName(const std::shared_ptr<iir::FieldAccessExpr>& expr,
                                     const iir::Interval& interval) {
  // TODO: Does not support unstructured grids right now
  return makeOnTheFlyFunctionCandidateName(expr, interval) + "_" +
         to_string(ast::cartesian, expr->getOffset(), "_", [](std::string const& name, int offset) {
           return name + "_" + ((offset < 0) ? "minus" : "") + std::to_string(std::abs(offset));
         });
}

std::string makeOnTheFlyFunctionCandidateName(const std::string fieldName,
                                              const sir::Interval& interval) {
  return fieldName + "_OnTheFly_" + sirIntervalToInterval(interval).toStringGen();
}

/// @brief visitor that will detect assignment (i.e. computations) to a temporary,
/// it will create a sir::StencilFunction out of this computation, and replace the assignment
/// expression in the AST by a NOExpr.
class TmpAssignment : public iir::ASTVisitorPostOrder, public NonCopyable {
protected:
  const iir::StencilMetaInformation& metadata_;
  sir::Interval interval_; // interval where the function declaration will be defined
  std::shared_ptr<sir::StencilFunction>
      tmpFunction_;            // sir function with the declaration of the tmp computation
  std::vector<int> accessIDs_; // accessIDs of the accesses that form the tmp = ... expression, that
                               // will become arguments of the stencil fn

  std::shared_ptr<iir::FieldAccessExpr> tmpFieldAccessExpr_ =
      nullptr; // the field access expr of the temporary that is captured and being replaced by
               // stencil fn
  const std::set<int>&
      skipAccessIDsOfMS_; // list of ids that will be skipped, since they dont fulfil the
                          // requirements, like they contain cycle dependencies, etc

public:
  TmpAssignment(const iir::StencilMetaInformation& metadata, sir::Interval const& interval,
                const std::set<int>& skipAccessIDsOfMS)
      : metadata_(metadata), interval_(interval), tmpFunction_(nullptr),
        skipAccessIDsOfMS_(skipAccessIDsOfMS) {}

  virtual ~TmpAssignment() {}

  const std::vector<int>& getAccessIDs() const { return accessIDs_; }

  std::shared_ptr<iir::FieldAccessExpr> getTemporaryFieldAccessExpr() {
    return tmpFieldAccessExpr_;
  }

  std::shared_ptr<sir::StencilFunction> temporaryStencilFunction() { return tmpFunction_; }

  bool foundTemporaryToReplace() { return (tmpFunction_ != nullptr); }

  /// @brief pre visit the node. The assignment expr visit will only continue processing the visitor
  /// for the right hand side of the =. Therefore all accesses capture here correspond to arguments
  /// for computing the tmp. They are captured as arguments of the stencil function being created
  virtual bool preVisitNode(std::shared_ptr<iir::FieldAccessExpr> const& expr) override {
#if DAWN_USING_ASSERTS
    DAWN_ASSERT(tmpFunction_);
    for(int idx : expr->getArgumentMap()) {
      DAWN_ASSERT(idx == -1);
    }
    for(int off : expr->getArgumentOffset())
      DAWN_ASSERT(off == 0);
#endif

    // record the field access as an argument to the stencil funcion
    if(!tmpFunction_->hasArg(expr->getName()) && expr != tmpFieldAccessExpr_) {

      int genLineKey = static_cast<std::underlying_type<SourceLocation::ReservedSL>::type>(
          SourceLocation::ReservedSL::Generated);
      sir::FieldDimensions&& dims =
          metadata_.getFieldDimensions(*expr->getData<iir::IIRAccessExprData>().AccessID);
      tmpFunction_->Args.push_back(std::make_shared<sir::Field>(
          expr->getName(), std::move(dims), SourceLocation(genLineKey, genLineKey)));

      accessIDs_.push_back(iir::getAccessID(expr));
    }
    // continue traversing
    return true;
  }

  virtual bool preVisitNode(std::shared_ptr<iir::VarAccessExpr> const& expr) override {
    DAWN_ASSERT(tmpFunction_);
    if(!metadata_.isAccessType(iir::FieldAccessType::GlobalVariable, iir::getAccessID(expr))) {
      // record the var access as an argument to the stencil funcion
      dawn_unreachable_internal("All the var access should have been promoted to temporaries");
    }
    return true;
  }

  virtual bool preVisitNode(std::shared_ptr<iir::VarDeclStmt> const& stmt) override {
    // a vardecl is assigning to a local variable, since the local variable promotion did not
    // promoted this to a tmp, we assume here the rules for tmp->function replacement were not
    // fulfil
    return false;
  }

  /// @brief capture a tmp computation
  virtual bool preVisitNode(std::shared_ptr<iir::AssignmentExpr> const& expr) override {
    if(isa<iir::FieldAccessExpr>(*(expr->getLeft()))) {
      // return and stop traversing the AST if the left hand of the =  is not a temporary
      int accessID = iir::getAccessID(expr->getLeft());
      if(skipAccessIDsOfMS_.count(accessID)) {
        return false;
      }
      tmpFieldAccessExpr_ = std::dynamic_pointer_cast<iir::FieldAccessExpr>(expr->getLeft());

      // otherwise we create a new stencil function
      std::string tmpFieldName = metadata_.getFieldNameFromAccessID(accessID);
      tmpFunction_ = std::make_shared<sir::StencilFunction>();

      tmpFunction_->Name = makeOnTheFlyFunctionCandidateName(tmpFieldName, interval_);
      tmpFunction_->Loc = expr->getSourceLocation();
      tmpFunction_->Intervals.push_back(std::make_shared<sir::Interval>(interval_));

      return true;
    }
    return false;
  }
  ///@brief once the "tmp= fn(args)" has been captured, the new stencil function to compute the
  /// temporary is finalized and the assignment is replaced by a NOExpr
  virtual std::shared_ptr<iir::Expr>
  postVisitNode(std::shared_ptr<iir::AssignmentExpr> const& expr) override {
    if(isa<iir::FieldAccessExpr>(*(expr->getLeft()))) {
      DAWN_ASSERT(tmpFieldAccessExpr_);
      const int accessID = iir::getAccessID(tmpFieldAccessExpr_);
      if(!metadata_.isAccessType(iir::FieldAccessType::StencilTemporary, accessID))
        return expr;

      DAWN_ASSERT(tmpFunction_);

      auto functionExpr = expr->getRight()->clone();
      auto retStmt = iir::makeReturnStmt(functionExpr);

      std::shared_ptr<iir::BlockStmt> root = iir::makeBlockStmt();
      root->push_back(retStmt);
      std::shared_ptr<iir::AST> ast = std::make_shared<iir::AST>(root);
      tmpFunction_->Asts.push_back(ast);

      return std::make_shared<iir::NOPExpr>();
    }
    return expr;
  }
};

/// @brief visitor that will capture all read accesses to the temporary. The offset used to access
/// the temporary is extracted and pass to all the stencil function arguments. A new stencil
/// function instantiation is then created and the field access expression replaced by a stencil
/// function call
class TmpReplacement : public iir::ASTVisitorPostOrder, public NonCopyable {
  // struct to store the stencil function instantiation of each parameter of a stencil function that
  // is at the same time instantiated as a stencil function
  struct NestedStencilFunctionArgs {
    int currentPos_ = 0;
    std::unordered_map<int, std::shared_ptr<iir::StencilFunctionInstantiation>> stencilFun_;
  };

protected:
  const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation_;
  iir::StencilMetaInformation& metadata_;
  OptimizerContext& context_;
  std::unordered_map<int, TemporaryFunctionProperties> const& temporaryFieldAccessIDToFunctionCall_;
  const iir::Interval interval_;
  const sir::Interval sirInterval_;
  const std::vector<ast::StencilCall*>& stackTrace_;
  std::shared_ptr<iir::Expr> skip_;

  // the prop of the arguments of nested stencil functions
  std::stack<bool> replaceInNestedFun_;

  unsigned int numTmpReplaced_ = 0;

  std::unordered_map<std::shared_ptr<iir::FieldAccessExpr>,
                     std::shared_ptr<iir::StencilFunctionInstantiation>>
      tmpToStencilFunctionMap_;

public:
  TmpReplacement(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                 OptimizerContext& context,
                 std::unordered_map<int, TemporaryFunctionProperties> const&
                     temporaryFieldAccessIDToFunctionCall,
                 const iir::Interval& interval, const std::vector<ast::StencilCall*>& stackTrace)
      : stencilInstantiation_(stencilInstantiation), metadata_(stencilInstantiation->getMetaData()),
        context_(context),
        temporaryFieldAccessIDToFunctionCall_(temporaryFieldAccessIDToFunctionCall),
        interval_(interval), sirInterval_(intervalToSIRInterval(interval)),
        stackTrace_(stackTrace) {}

  virtual ~TmpReplacement() {}

  unsigned int getNumTmpReplaced() { return numTmpReplaced_; }
  void resetNumTmpReplaced() { numTmpReplaced_ = 0; }

  virtual bool preVisitNode(std::shared_ptr<iir::StencilFunCallExpr> const& expr) override {
    // we check which arguments of the stencil function are also a stencil function call
    bool doReplaceTmp = false;
    for(auto arg : expr->getArguments()) {
      if(isa<iir::FieldAccessExpr>(*arg)) {
        int accessID = iir::getAccessID(arg);
        if(temporaryFieldAccessIDToFunctionCall_.count(accessID)) {
          doReplaceTmp = true;
        }
      }
    }
    if(doReplaceTmp)
      replaceInNestedFun_.push(true);
    else
      replaceInNestedFun_.push(false);

    return true;
  }

  virtual std::shared_ptr<iir::Expr>
  postVisitNode(std::shared_ptr<iir::StencilFunCallExpr> const& expr) override {
    // at the post visit of a stencil function node, we will replace the arguments to "tmp" fields
    // by stecil function calls
    std::shared_ptr<iir::StencilFunctionInstantiation> thisStencilFun =
        metadata_.getStencilFunctionInstantiation(expr);

    if(!replaceInNestedFun_.top())
      return expr;

    // we need to remove the previous stencil function that had "tmp" field as argument from the
    // registry, before we replace it with a StencilFunCallExpr (that computes "tmp") argument
    metadata_.deregisterStencilFunction(thisStencilFun);
    // reset the use of nested function calls to continue using the visitor
    replaceInNestedFun_.pop();

    return expr;
  }

  /// @brief previsit the access to a temporary. Finalize the stencil function instantiation and
  /// recompute its accesses
  virtual bool preVisitNode(std::shared_ptr<iir::AssignmentExpr> const& expr) override {
    // we would like to identify fields that are lhs of an assignment expr, so that we skip them and
    // dont replace them
    if(isa<iir::FieldAccessExpr>(expr->getLeft().get())) {
      skip_ = expr->getLeft();
    }
    return true;
  }

  bool replaceFieldByFunction(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
    int accessID = iir::getAccessID(expr);
    if(!temporaryFieldAccessIDToFunctionCall_.count(accessID)) {
      return false;
    }
    const auto& tempFuncProperties = temporaryFieldAccessIDToFunctionCall_.at(accessID);

    return (expr != skip_) && tempFuncProperties.interval_.contains(interval_);
  }

  /// @brief previsit the access to a temporary. Finalize the stencil function instantiation and
  /// recompute its accesses
  virtual bool preVisitNode(std::shared_ptr<iir::FieldAccessExpr> const& expr) override {
    int accessID = iir::getAccessID(expr);

    if(!replaceFieldByFunction(expr)) {
      return true;
    }

    const auto& tempFuncProperties = temporaryFieldAccessIDToFunctionCall_.at(accessID);

    // TODO we need to version to tmp function generation, in case tmp is recomputed multiple
    // times
    std::string callee = makeOnTheFlyFunctionCandidateName(expr, interval_);
    std::shared_ptr<iir::StencilFunctionInstantiation> stencilFun =
        metadata_.getStencilFunctionInstantiationCandidate(callee, interval_);

    std::string fnClone = makeOnTheFlyFunctionName(expr, interval_);

    // retrieve the sir stencil function definition
    std::shared_ptr<sir::StencilFunction> sirStencilFunction =
        tempFuncProperties.sirStencilFunction_;

    // we create a new sir stencil function, since its name is demangled from the offsets.
    // for example, for a tmp(i+1) the stencil function is named as tmp_OnTheFly_iplus1
    std::shared_ptr<sir::StencilFunction> sirStencilFunctionInstance =
        std::make_shared<sir::StencilFunction>(*sirStencilFunction);

    sirStencilFunctionInstance->Name = fnClone;

    // TODO is this really needed, we only change the name, can we map multiple function
    // instantiations (i.e. different offsets) to the same sir stencil function
    // insert the sir::stencilFunction into the StencilInstantiation
    stencilInstantiation_->getIIR()->insertStencilFunction(sirStencilFunctionInstance);

    // we clone the stencil function instantiation of the candidate so that each instance of the st
    // function has its own private copy of the expressions (i.e. ast)
    std::shared_ptr<iir::StencilFunctionInstantiation> cloneStencilFun =
        metadata_.cloneStencilFunctionCandidate(stencilFun, fnClone);

    auto& accessIDsOfArgs = tempFuncProperties.accessIDArgs_;

    // here we create the arguments of the stencil function instantiation.
    // find the accessID of all args, and create a new FieldAccessExpr with an offset
    // corresponding
    // to the offset used to access the temporary
    for(auto accessID_ : (accessIDsOfArgs)) {
      std::shared_ptr<iir::FieldAccessExpr> arg = std::make_shared<iir::FieldAccessExpr>(
          metadata_.getFieldNameFromAccessID(accessID_), expr->getOffset());
      cloneStencilFun->getExpression()->insertArgument(arg);

      arg->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(accessID_);
    }

#if DAWN_USING_ASSERTS
    for(int idx : expr->getArgumentMap()) {
      DAWN_ASSERT(idx == -1);
    }
    for(int off : expr->getArgumentOffset())
      DAWN_ASSERT(off == 0);
#endif

    const auto& argToAccessIDMap = stencilFun->ArgumentIndexToCallerAccessIDMap();
    for(auto pair : argToAccessIDMap) {
      int accessID_ = pair.second;
      cloneStencilFun->setCallerInitialOffsetFromAccessID(accessID_, expr->getOffset());
    }

    metadata_.finalizeStencilFunctionSetup(cloneStencilFun);
    std::unordered_map<std::string, int> fieldsMap;

    const auto& arguments = cloneStencilFun->getArguments();
    for(std::size_t argIdx = 0; argIdx < arguments.size(); ++argIdx) {
      if(sir::Field* field = dyn_cast<sir::Field>(arguments[argIdx].get())) {
        int AccessID = cloneStencilFun->getCallerAccessIDOfArgField(argIdx);
        fieldsMap[field->Name] = AccessID;
      }
    }

    // recompute the list of <statement, accesses> pairs
    StatementMapper statementMapper(stencilInstantiation_.get(), context_, stackTrace_,
                                    *(cloneStencilFun->getDoMethod()), interval_, fieldsMap,
                                    cloneStencilFun);

    cloneStencilFun->getAST()->accept(statementMapper);

    // final checks
    cloneStencilFun->checkFunctionBindings();

    // register the FieldAccessExpr -> StencilFunctionInstantation into a map for future
    // replacement
    // of the node in the post visit
    DAWN_ASSERT(!tmpToStencilFunctionMap_.count(expr));
    tmpToStencilFunctionMap_[expr] = cloneStencilFun;

    return true;
  }

  /// @brief replace the access to a temporary by a stencil function call expression
  virtual std::shared_ptr<iir::Expr>
  postVisitNode(std::shared_ptr<iir::FieldAccessExpr> const& expr) override {
    // if the field access is not identified as a temporary being replaced just return without
    // substitution
    if(!replaceFieldByFunction(expr)) {
      return expr;
    }

    // TODO we need to version to tmp function generation, in case tmp is recomputed multiple
    // times
    std::string callee = makeOnTheFlyFunctionName(expr, interval_);

    DAWN_ASSERT(tmpToStencilFunctionMap_.count(expr));

    auto stencilFunCall = tmpToStencilFunctionMap_.at(expr)->getExpression();

    numTmpReplaced_++;
    return stencilFunCall;
  }
};

} // anonymous namespace

PassTemporaryToStencilFunction::PassTemporaryToStencilFunction(OptimizerContext& context)
    : Pass(context, "PassTemporaryToStencilFunction") {}

SkipIDs PassTemporaryToStencilFunction::computeSkipAccessIDs(
    const std::unique_ptr<iir::Stencil>& stencilPtr,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) const {

  const auto& metadata = stencilInstantiation->getMetaData();
  SkipIDs skipIDs;
  // Iterate multi-stages backwards in order to identify local variables that need to be promoted
  // to temporaries
  for(const auto& multiStage : stencilPtr->getChildren()) {
    iir::DependencyGraphAccesses graph(stencilInstantiation->getMetaData());
    for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*multiStage)) {
      for(const auto& stmt : doMethod->getAST().getStatements()) {
        graph.insertStatement(stmt);
      }
    }
    // TODO this is crashing for the divergene helper
    //    graph.toDot("PP");

    // all the fields with self-dependencies are discarded, e.g. w += w[k+1]
    skipIDs.insertAccessIDsOfMS(multiStage->getID(), graph.computeIDsWithCycles());
    for(const auto& fieldPair : multiStage->getFields()) {
      const auto& field = fieldPair.second;

      // we dont consider non temporary fields
      if(!metadata.isAccessType(iir::FieldAccessType::StencilTemporary, field.getAccessID())) {
        skipIDs.appendAccessIDsToMS(multiStage->getID(), field.getAccessID());
        continue;
      }
      // The scope of the temporary has to be a MS.
      // TODO Note the algorithm is not mathematically
      // complete here. We need to make sure that first access is always a write
      if(field.getIntend() != iir::Field::IntendKind::InputOutput) {
        skipIDs.appendAccessIDsToMS(multiStage->getID(), field.getAccessID());
        continue;
      }
      // we require that there are no vertical extents, otherwise the definition of a tmp might be
      // in a different interval than where it is used
      auto extents = field.getExtents();
      if(!extents.isVerticalPointwise()) {
        skipIDs.appendAccessIDsToMS(multiStage->getID(), field.getAccessID());
        continue;
      }
      if(extents.isHorizontalPointwise()) {
        skipIDs.appendAccessIDsToMS(multiStage->getID(), field.getAccessID());
        continue;
      }
    }
  }

  return skipIDs;
}

bool PassTemporaryToStencilFunction::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  const auto& metadata = stencilInstantiation->getMetaData();

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    const auto& fields = stencilPtr->getFields();

    SkipIDs skipIDs = computeSkipAccessIDs(stencilPtr, stencilInstantiation);

    std::unordered_set<int> localVarAccessIDs;
    LocalVariablePromotion localVariablePromotion(metadata, *stencilPtr, fields, skipIDs,
                                                  localVarAccessIDs);

    for(auto multiStageIt = stencilPtr->childrenRBegin();
        multiStageIt != stencilPtr->childrenREnd(); ++multiStageIt) {

      for(auto stageIt = (*multiStageIt)->childrenRBegin();
          stageIt != (*multiStageIt)->childrenREnd(); ++stageIt) {

        for(auto doMethodIt = (*stageIt)->childrenRBegin();
            doMethodIt != (*stageIt)->childrenREnd(); doMethodIt++) {
          for(auto stmtIt = (*doMethodIt)->getAST().getStatements().rbegin();
              stmtIt != (*doMethodIt)->getAST().getStatements().rend(); stmtIt++) {

            (*stmtIt)->acceptAndReplace(localVariablePromotion);
          }
        }
      }
    }

    // perform the promotion "local var"->temporary
    for(auto varID : localVarAccessIDs) {
      if(metadata.isAccessType(iir::FieldAccessType::GlobalVariable, varID))
        continue;

      promoteLocalVariableToTemporaryField(stencilInstantiation.get(), stencilPtr.get(), varID,
                                           stencilPtr->getLifetime(varID),
                                           iir::TemporaryScope::StencilTemporary);
    }

    skipIDs = computeSkipAccessIDs(stencilPtr, stencilInstantiation);

    // Iterate multi-stages for the replacement of temporaries by stencil functions
    for(const auto& multiStage : stencilPtr->getChildren()) {
      auto multiInterval = multiStage->computePartitionOfIntervals();
      for(const auto& interval : multiInterval.getIntervals()) {

        auto skipAccessIDsOfMS = skipIDs.accessIDs.at(multiStage->getID());

        std::unordered_map<int, TemporaryFunctionProperties> temporaryFieldExprToFunction;

        for(const auto& stagePtr : multiStage->getChildren()) {
          bool isATmpReplaced = false;
          for(const auto& doMethodPtr : stagePtr->getChildren()) {
            if(!doMethodPtr->getInterval().overlaps(interval)) {
              continue;
            }

            for(const auto& stmt : doMethodPtr->getAST().getStatements()) {

              DAWN_ASSERT((stmt->getKind() != iir::Stmt::Kind::ReturnStmt) &&
                          (stmt->getKind() != iir::Stmt::Kind::StencilCallDeclStmt) &&
                          (stmt->getKind() != iir::Stmt::Kind::VerticalRegionDeclStmt) &&
                          (stmt->getKind() != iir::Stmt::Kind::BoundaryConditionDeclStmt));

              // We exclude blocks or If/Else stmt
              if((stmt->getKind() != iir::Stmt::Kind::ExprStmt) &&
                 (stmt->getKind() != iir::Stmt::Kind::VarDeclStmt)) {
                continue;
              }

              {
                // TODO catch a temp expr
                const iir::Interval& doMethodInterval = doMethodPtr->getInterval();
                const sir::Interval sirInterval = intervalToSIRInterval(interval);

                DAWN_ASSERT(stmt->getData<iir::IIRStmtData>().StackTrace);

                // run the replacer visitor
                TmpReplacement tmpReplacement(stencilInstantiation, context_,
                                              temporaryFieldExprToFunction, interval,
                                              *stmt->getData<iir::IIRStmtData>().StackTrace);
                stmt->acceptAndReplace(tmpReplacement);

                // flag if a least a tmp has been replaced within this stage
                isATmpReplaced = isATmpReplaced || (tmpReplacement.getNumTmpReplaced() != 0);

                if(tmpReplacement.getNumTmpReplaced() != 0) {

                  iir::DoMethod tmpStmtDoMethod(doMethodInterval, metadata);

                  StatementMapper statementMapper(
                      stencilInstantiation.get(), context_,
                      *stmt->getData<iir::IIRStmtData>().StackTrace, tmpStmtDoMethod, sirInterval,
                      stencilInstantiation->getMetaData().getNameToAccessIDMap(), nullptr);

                  std::shared_ptr<iir::BlockStmt> blockStmt =
                      iir::makeBlockStmt(std::vector<std::shared_ptr<iir::Stmt>>{stmt});
                  blockStmt->accept(statementMapper);

                  DAWN_ASSERT(tmpStmtDoMethod.getAST().getStatements().size() == 1);

                  const std::shared_ptr<iir::Stmt>& replacementStmt =
                      *(tmpStmtDoMethod.getAST().getStatements().begin());
                  computeAccesses(stencilInstantiation->getMetaData(), replacementStmt);

                  doMethodPtr->getAST().replaceChildren(stmt, replacementStmt);
                  doMethodPtr->update(iir::NodeUpdateType::level);
                }

                // find patterns like tmp = fn(args)...;
                TmpAssignment tmpAssignment(metadata, sirInterval, skipAccessIDsOfMS);
                stmt->acceptAndReplace(tmpAssignment);
                if(tmpAssignment.foundTemporaryToReplace()) {
                  std::shared_ptr<sir::StencilFunction> stencilFunction =
                      tmpAssignment.temporaryStencilFunction();
                  std::shared_ptr<iir::AST> ast = stencilFunction->getASTOfInterval(sirInterval);

                  DAWN_ASSERT(ast);
                  DAWN_ASSERT(stencilFunction);

                  std::shared_ptr<iir::StencilFunCallExpr> stencilFunCallExpr =
                      std::make_shared<iir::StencilFunCallExpr>(stencilFunction->Name);

                  // all the temporary computations captured are stored in this map of <ID, tmp
                  // properties>
                  // for later use of the replacer visitor
                  const int accessID =
                      iir::getAccessID(tmpAssignment.getTemporaryFieldAccessExpr());

                  if(!temporaryFieldExprToFunction.count(accessID)) {
                    temporaryFieldExprToFunction.emplace(
                        accessID,
                        TemporaryFunctionProperties{
                            stencilFunCallExpr, tmpAssignment.getAccessIDs(), stencilFunction,
                            tmpAssignment.getTemporaryFieldAccessExpr(), doMethodInterval});
                  } else {
                    // emplace_and_assign is available only with c++17
                    temporaryFieldExprToFunction.erase(accessID);
                    temporaryFieldExprToFunction.emplace(
                        accessID,
                        TemporaryFunctionProperties{
                            stencilFunCallExpr, tmpAssignment.getAccessIDs(), stencilFunction,
                            tmpAssignment.getTemporaryFieldAccessExpr(), doMethodInterval});
                  }

                  // first instantiation of the stencil function that is inserted in the IIR as a
                  // candidate stencil function
                  // notice we clone the ast, so that every stencil function instantiation has a
                  // private copy of the ast (so that it can be transformed). However that is not
                  // enough since this function is inserted as a candidate, and a candidate can be
                  // inserted as multiple st function instances. Later when the candidate
                  // is finalized in a concrete instance, the ast will have to be cloned again
                  ast = ast->clone();
                  auto stencilFun = stencilInstantiation->makeStencilFunctionInstantiation(
                      stencilFunCallExpr, stencilFunction, ast, sirInterval, nullptr);

                  int argID = 0;
                  for(auto accessID_ : tmpAssignment.getAccessIDs()) {
                    stencilFun->setCallerAccessIDOfArgField(argID++, accessID_);
                  }
                }
              }
            }
          }
          if(isATmpReplaced) {
            stagePtr->update(iir::NodeUpdateType::level);
          }
        }

        std::ostringstream ss;
        for(auto tmpFieldPair : temporaryFieldExprToFunction) {
          int accessID = tmpFieldPair.first;
          auto tmpProperties = tmpFieldPair.second;
          ss << " [ replace tmp:" << metadata.getFieldNameFromAccessID(accessID)
             << "; line : " << tmpProperties.tmpFieldAccessExpr_->getSourceLocation().Line << " ] ";
        }
        if(temporaryFieldExprToFunction.empty())
          ss << "no replacement found";
        DAWN_LOG(INFO) << stencilInstantiation->getName() << ss.str();
      }
    }
    // eliminate empty stages or stages with only NOPExpr statements
    stencilPtr->childrenEraseIf(
        [](const std::unique_ptr<iir::MultiStage>& m) -> bool { return m->isEmptyOrNullStmt(); });
    for(const auto& multiStage : stencilPtr->getChildren()) {
      multiStage->childrenEraseIf(
          [](const std::unique_ptr<iir::Stage>& s) -> bool { return s->isEmptyOrNullStmt(); });
    }
    for(const auto& multiStage : stencilPtr->getChildren()) {
      multiStage->update(iir::NodeUpdateType::levelAndTreeAbove);
    }
  }

  return true;
}

} // namespace dawn
