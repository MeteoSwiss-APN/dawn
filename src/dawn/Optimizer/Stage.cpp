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

#include "dawn/Optimizer/Stage.h"
#include "dawn/Optimizer/AccessUtils.h"
#include "dawn/Optimizer/DependencyGraphAccesses.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/Support/Logging.h"
#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>

namespace dawn {

Stage::Stage(StencilInstantiation& context, MultiStage* multiStage, int StageID,
             const Interval& interval)
    : stencilInstantiation_(context), multiStage_(multiStage), StageID_(StageID), extents_{} {
  DoMethods_.emplace_back(make_unique<DoMethod>(this, interval));
}

std::vector<std::unique_ptr<DoMethod>>& Stage::getDoMethods() { return DoMethods_; }

const std::vector<std::unique_ptr<DoMethod>>& Stage::getDoMethods() const { return DoMethods_; }

bool Stage::hasSingleDoMethod() const { return (DoMethods_.size() == 1); }

DoMethod& Stage::getSingleDoMethod() {
  DAWN_ASSERT_MSG(hasSingleDoMethod(), "stage contains multiple Do-Methods");
  return *DoMethods_.front();
}

const DoMethod& Stage::getSingleDoMethod() const {
  DAWN_ASSERT_MSG(hasSingleDoMethod(), "stage contains multiple Do-Methods");
  return *DoMethods_.front();
}

boost::optional<Interval> Stage::computeEnclosingAccessInterval(const int accessID) const {
  boost::optional<Interval> interval;
  for(auto const& doMethod : DoMethods_) {
    boost::optional<Interval> doInterval = doMethod->computeEnclosingAccessInterval(accessID);

    if(doInterval) {
      if(interval)
        (*interval).merge(*doInterval);
      else
        interval = doInterval;
    }
  }
  return interval;
}

std::vector<Interval> Stage::getIntervals() const {
  std::vector<Interval> intervals;
  std::transform(
      DoMethods_.begin(), DoMethods_.end(), std::back_inserter(intervals),
      [&](const std::unique_ptr<DoMethod>& doMethod) { return doMethod->getInterval(); });
  return intervals;
}

Interval Stage::getEnclosingInterval() const {
  Interval interval = DoMethods_.front()->getInterval();
  for(int i = 1; i < DoMethods_.size(); ++i)
    interval.merge(DoMethods_[i]->getInterval());
  return interval;
}

Extent Stage::getMaxVerticalExtent() const {
  Extent verticalExtent;
  std::for_each(fields_.begin(), fields_.end(),
                [&](const Field& field) { verticalExtent.merge(field.getExtents()[2]); });
  return verticalExtent;
}

Interval Stage::getEnclosingExtendedInterval() const {
  return getEnclosingInterval().extendInterval(getMaxVerticalExtent());
}

bool Stage::overlaps(const Stage& other) const {
  // This is a more conservative test.. if it fails we are certain nothing overlaps
  if(!getEnclosingExtendedInterval().overlaps(other.getEnclosingExtendedInterval()))
    return false;

  for(const auto& otherDoMethodPtr : other.getDoMethods())
    if(overlaps(otherDoMethodPtr->getInterval(), other.getFields()))
      return true;
  return false;
}

bool Stage::overlaps(const Interval& interval, ArrayRef<Field> fields) const {
  for(const auto& doMethodPtr : DoMethods_) {
    const Interval& thisInterval = doMethodPtr->getInterval();

    for(const Field& thisField : getFields()) {
      for(const Field& field : fields) {
        if(thisField.getAccessID() != field.getAccessID())
          continue;

        if(thisInterval.extendInterval(thisField.getExtents()[2])
               .overlaps(interval.extendInterval(field.getExtents()[2])))
          return true;
      }
    }
  }

  return false;
}

///
/// @brief The CaptureStencilFunctionCallGlobalParams class
/// is an AST visitor used to capture accesses to global accessor from within
/// stencil functions called from a stage
class CaptureStencilFunctionCallGlobalParams : public ASTVisitorForwarding {

  std::unordered_set<int>& globalVariables_;
  StencilFunctionInstantiation* currentFunction_;
  StencilInstantiation& stencilInstantiation_;
  std::shared_ptr<const StencilFunctionInstantiation> function_;

public:
  CaptureStencilFunctionCallGlobalParams(std::unordered_set<int>& globalVariables,
                                         StencilInstantiation& stencilInstantiation)
      : globalVariables_(globalVariables), stencilInstantiation_(stencilInstantiation),
        function_(nullptr) {}

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    // Find the referenced stencil function
    std::shared_ptr<const StencilFunctionInstantiation> stencilFun =
        function_ ? function_->getStencilFunctionInstantiation(expr)
                  : stencilInstantiation_.getStencilFunctionInstantiation(expr);

    DAWN_ASSERT(stencilFun);
    for(auto it : stencilFun->getAccessIDSetGlobalVariables()) {
      globalVariables_.insert(it);
    }
    std::shared_ptr<const StencilFunctionInstantiation> prevFunction_ = function_;
    function_ = stencilFun;
    stencilFun->getAST()->accept(*this);
    function_ = prevFunction_;
  }
};

void Stage::update() {
  fields_.clear();
  globalVariables_.clear();
  globalVariablesFromStencilFunctionCalls_.clear();
  allGlobalVariables_.clear();

  // Compute the fields and their intended usage. Fields can be in one of three states: `Output`,
  // `InputOutput` or `Input` which implements the following state machine:
  //
  //    +-------+                               +--------+
  //    | Input |                               | Output |
  //    +-------+                               +--------+
  //        |                                       |
  //        |            +-------------+            |
  //        +----------> | InputOutput | <----------+
  //                     +-------------+
  //
  std::unordered_map<int, Field> inputOutputFields;
  std::unordered_map<int, Field> inputFields;
  std::unordered_map<int, Field> outputFields;

  CaptureStencilFunctionCallGlobalParams functionCallGlobaParamVisitor(
      globalVariablesFromStencilFunctionCalls_, stencilInstantiation_);
  for(const auto& doMethodPtr : DoMethods_) {
    const DoMethod& doMethod = *doMethodPtr;
    for(const auto& statementAccessesPair : doMethod.getStatementAccessesPairs()) {
      statementAccessesPair->getStatement()->ASTStmt->accept(functionCallGlobaParamVisitor);
      const auto& access = statementAccessesPair->getAccesses();
      DAWN_ASSERT(access);

      for(const auto& accessPair : access->getWriteAccesses()) {
        int AccessID = accessPair.first;

        // Does this AccessID correspond to a field access?
        if(!stencilInstantiation_.isField(AccessID)) {
          if(stencilInstantiation_.isGlobalVariable(AccessID))
            globalVariables_.insert(AccessID);
          continue;
        }

        AccessUtils::recordWriteAccess(inputOutputFields, inputFields, outputFields, AccessID,
                                       doMethod.getInterval());
      }

      for(const auto& accessPair : access->getReadAccesses()) {
        int AccessID = accessPair.first;

        // Does this AccessID correspond to a field access?
        if(!stencilInstantiation_.isField(AccessID)) {
          if(stencilInstantiation_.isGlobalVariable(AccessID))
            globalVariables_.insert(AccessID);
          continue;
        }

        AccessUtils::recordReadAccess(inputOutputFields, inputFields, outputFields, AccessID,
                                      doMethod.getInterval());
      }

      const std::shared_ptr<Statement> statement = statementAccessesPair->getStatement();
      DAWN_ASSERT(statement);
      DAWN_ASSERT(statement->ASTStmt);

      // capture all the accesses to global accesses of stencil function called
      // from this statement
      statement->ASTStmt->accept(functionCallGlobaParamVisitor);
    }
  }

  allGlobalVariables_.insert(globalVariables_.begin(), globalVariables_.end());
  allGlobalVariables_.insert(globalVariablesFromStencilFunctionCalls_.begin(),
                             globalVariablesFromStencilFunctionCalls_.end());

  // Merge inputFields, outputFields and fields
  for(auto fieldPair : outputFields)
    fields_.push_back(fieldPair.second);

  for(auto fieldPair : inputOutputFields)
    fields_.push_back(fieldPair.second);

  for(auto fieldPair : inputFields)
    fields_.push_back(fieldPair.second);

  if(fields_.empty()) {
    DAWN_LOG(WARNING) << "no fields referenced in stage";
    return;
  }

  // Index to speedup lookup into fields map
  std::unordered_map<int, std::vector<Field>::iterator> AccessIDToFieldMap;
  for(auto it = fields_.begin(), end = fields_.end(); it != end; ++it)
    AccessIDToFieldMap.insert(std::make_pair(it->getAccessID(), it));

  // Compute the extents of each field by accumulating the extents of each access to field in the
  // stage
  for(const auto& doMethodPtr : DoMethods_) {
    const DoMethod& doMethod = *doMethodPtr;

    for(const auto& statementAccessesPair : doMethod.getStatementAccessesPairs()) {
      const auto& access = statementAccessesPair->getAccesses();

      // first => AccessID, second => Extent
      for(auto& accessPair : access->getWriteAccesses()) {
        if(!stencilInstantiation_.isField(accessPair.first))
          continue;

        AccessIDToFieldMap[accessPair.first]->mergeExtents(accessPair.second);
      }

      for(const auto& accessPair : access->getReadAccesses()) {
        if(!stencilInstantiation_.isField(accessPair.first))
          continue;

        AccessIDToFieldMap[accessPair.first]->mergeExtents(accessPair.second);
      }
    }
  }
}

bool Stage::hasGlobalVariables() const {
  return (!globalVariables_.empty()) || (!globalVariablesFromStencilFunctionCalls_.empty());
}

const std::unordered_set<int>& Stage::getGlobalVariables() const { return globalVariables_; }

const std::unordered_set<int>& Stage::getGlobalVariablesFromStencilFunctionCalls() const {
  return globalVariablesFromStencilFunctionCalls_;
}

const std::unordered_set<int>& Stage::getAllGlobalVariables() const { return allGlobalVariables_; }

void Stage::addDoMethod(std::unique_ptr<DoMethod>& doMethod) {
  DAWN_ASSERT_MSG(std::find_if(DoMethods_.begin(), DoMethods_.end(),
                               [&](const std::unique_ptr<DoMethod>& doMethodPtr) {
                                 return doMethodPtr->getInterval() == doMethod->getInterval();
                               }) == DoMethods_.end(),
                  "Do-Method with given interval already exists!");
  DoMethods_.emplace_back(std::move(doMethod));
  update();
}

void Stage::appendDoMethod(std::unique_ptr<DoMethod>& from, std::unique_ptr<DoMethod>& to,
                           const std::shared_ptr<DependencyGraphAccesses>& dependencyGraph) {
  DAWN_ASSERT_MSG(std::find(DoMethods_.begin(), DoMethods_.end(), to) != DoMethods_.end(),
                  "'to' DoMethod does not exists");
  DAWN_ASSERT_MSG(from->getInterval() == to->getInterval(),
                  "DoMethods have incompatible intervals!");

  to->setDependencyGraph(dependencyGraph);
  to->getStatementAccessesPairs().insert(
      to->getStatementAccessesPairs().end(),
      std::make_move_iterator(from->getStatementAccessesPairs().begin()),
      std::make_move_iterator(from->getStatementAccessesPairs().end()));
  update();
}

LoopOrderKind Stage::getLoopOrder() const { return multiStage_->getLoopOrder(); }

std::vector<std::shared_ptr<Stage>>
Stage::split(std::deque<int>& splitterIndices,
             const std::deque<std::shared_ptr<DependencyGraphAccesses>>* graphs) {
  DAWN_ASSERT_MSG(hasSingleDoMethod(), "Stage::split does not support multiple Do-Methods");
  DoMethod& thisDoMethod = getSingleDoMethod();
  auto& thisStatementAccessesPairs = thisDoMethod.getStatementAccessesPairs();

  DAWN_ASSERT(thisStatementAccessesPairs.size() >= 2);
  DAWN_ASSERT(!graphs || splitterIndices.size() == graphs->size() - 1);

  std::vector<std::shared_ptr<Stage>> newStages;

  splitterIndices.push_back(thisStatementAccessesPairs.size() - 1);
  std::size_t prevSplitterIndex = 0;

  // Create new stages
  for(std::size_t i = 0; i < splitterIndices.size(); ++i) {
    std::size_t nextSplitterIndex = splitterIndices[i] + 1;

    newStages.push_back(std::make_shared<Stage>(stencilInstantiation_, multiStage_,
                                                stencilInstantiation_.nextUID(),
                                                thisDoMethod.getInterval()));
    Stage& newStage = *newStages.back();
    DoMethod& doMethod = newStage.getSingleDoMethod();

    if(graphs)
      doMethod.setDependencyGraph((*graphs)[i]);

    // The new stage contains the statements in the range [prevSplitterIndex , nextSplitterIndex)
    for(std::size_t idx = prevSplitterIndex; idx < nextSplitterIndex; ++idx)
      doMethod.getStatementAccessesPairs().emplace_back(std::move(thisStatementAccessesPairs[idx]));

    // Update the fields of the new stage
    newStage.update();

    prevSplitterIndex = nextSplitterIndex;
  }

  return newStages;
}

} // namespace dawn
