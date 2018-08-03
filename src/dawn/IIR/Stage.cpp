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

#include "dawn/IIR/Stage.h"
#include "dawn/Optimizer/AccessUtils.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/Support/Logging.h"
#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>

namespace dawn {
namespace iir {

Stage::Stage(StencilInstantiation& context, MultiStage* multiStage, int StageID,
             const Interval& interval)
    : stencilInstantiation_(context), multiStage_(multiStage), StageID_(StageID),
      extents_{0, 0, 0, 0, 0, 0} {
  insertChild(make_unique<DoMethod>(interval));
}

DoMethod& Stage::getSingleDoMethod() {
  DAWN_ASSERT_MSG(hasSingleDoMethod(), "stage contains multiple Do-Methods");
  return *(getChildren().front());
}

const DoMethod& Stage::getSingleDoMethod() const {
  DAWN_ASSERT_MSG(hasSingleDoMethod(), "stage contains multiple Do-Methods");
  return *(getChildren().front());
}

bool Stage::hasSingleDoMethod() const { return (children_.size() == 1); }

boost::optional<Interval>
Stage::computeEnclosingAccessInterval(const int accessID, const bool mergeWithDoInterval) const {
  boost::optional<Interval> interval;
  for(auto const& doMethod : getChildren()) {
    boost::optional<Interval> doInterval =
        doMethod->computeEnclosingAccessInterval(accessID, mergeWithDoInterval);

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
  std::transform(childrenBegin(), childrenEnd(), std::back_inserter(intervals),
                 [&](const DoMethodSmartPtr_t& doMethod) { return doMethod->getInterval(); });
  return intervals;
}

Interval Stage::getEnclosingInterval() const {
  Interval interval = getChildren().front()->getInterval();
  for(const auto& doMethod : getChildren())
    interval.merge(doMethod->getInterval());
  return interval;
}

Extent Stage::getMaxVerticalExtent() const {
  Extent verticalExtent;
  std::for_each(fields_.begin(), fields_.end(), [&](const std::pair<int, Field>& pair) {
    verticalExtent.merge(pair.second.getExtents()[2]);
  });
  return verticalExtent;
}

Interval Stage::getEnclosingExtendedInterval() const {
  return getEnclosingInterval().extendInterval(getMaxVerticalExtent());
}

bool Stage::overlaps(const Stage& other) const {
  // This is a more conservative test.. if it fails we are certain nothing overlaps
  if(!getEnclosingExtendedInterval().overlaps(other.getEnclosingExtendedInterval()))
    return false;

  for(const auto& otherDoMethodPtr : other.getChildren())
    if(overlaps(otherDoMethodPtr->getInterval(), other.getFields()))
      return true;
  return false;
}

bool Stage::overlaps(const Interval& interval, const std::unordered_map<int, Field>& fields) const {
  for(const auto& doMethodPtr : getChildren()) {
    const Interval& thisInterval = doMethodPtr->getInterval();

    for(const auto& thisFieldPair : getFields()) {
      const Field& thisField = thisFieldPair.second;
      for(const auto& fieldPair : fields) {
        const Field& field = fieldPair.second;
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
  for(const auto& doMethodPtr : getChildren()) {
    const DoMethod& doMethod = *doMethodPtr;
    for(const auto& statementAccessesPair : doMethod.getChildren()) {
      statementAccessesPair->getStatement()->ASTStmt->accept(functionCallGlobaParamVisitor);
      const auto& access = statementAccessesPair->getAccesses();
      DAWN_ASSERT(access);

      for(const auto& accessPair : access->getWriteAccesses()) {
        int AccessID = accessPair.first;
        Extents const& extents = accessPair.second;

        // Does this AccessID correspond to a field access?
        if(!stencilInstantiation_.isField(AccessID)) {
          if(stencilInstantiation_.isGlobalVariable(AccessID))
            globalVariables_.insert(AccessID);
          continue;
        }
        AccessUtils::recordWriteAccess(inputOutputFields, inputFields, outputFields, AccessID,
                                       extents, doMethod.getInterval());
      }

      for(const auto& accessPair : access->getReadAccesses()) {
        int AccessID = accessPair.first;
        Extents const& extents = accessPair.second;

        // Does this AccessID correspond to a field access?
        if(!stencilInstantiation_.isField(AccessID)) {
          if(stencilInstantiation_.isGlobalVariable(AccessID))
            globalVariables_.insert(AccessID);
          continue;
        }

        AccessUtils::recordReadAccess(inputOutputFields, inputFields, outputFields, AccessID,
                                      extents, doMethod.getInterval());
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
  fields_.insert(outputFields.begin(), outputFields.end());
  fields_.insert(inputOutputFields.begin(), inputOutputFields.end());
  fields_.insert(inputFields.begin(), inputFields.end());

  if(fields_.empty()) {
    DAWN_LOG(WARNING) << "no fields referenced in stage";
    return;
  }

  // Compute the extents of each field by accumulating the extents of each access to field in the
  // stage
  for(const auto& doMethodPtr : getChildren()) {
    const DoMethod& doMethod = *doMethodPtr;

    for(const auto& statementAccessesPair : doMethod.getChildren()) {
      const auto& access = statementAccessesPair->getAccesses();

      // first => AccessID, second => Extent
      for(auto& accessPair : access->getWriteAccesses()) {
        if(!stencilInstantiation_.isField(accessPair.first))
          continue;

        fields_.at(accessPair.first).mergeWriteExtents(accessPair.second);
      }

      for(const auto& accessPair : access->getReadAccesses()) {
        if(!stencilInstantiation_.isField(accessPair.first))
          continue;

        fields_.at(accessPair.first).mergeReadExtents(accessPair.second);
      }
    }
  }
  // TODO
  //  getParent()->update();
}

bool Stage::hasGlobalVariables() const {
  return (!globalVariables_.empty()) || (!globalVariablesFromStencilFunctionCalls_.empty());
}

const std::unordered_set<int>& Stage::getGlobalVariables() const { return globalVariables_; }

const std::unordered_set<int>& Stage::getGlobalVariablesFromStencilFunctionCalls() const {
  return globalVariablesFromStencilFunctionCalls_;
}

const std::unordered_set<int>& Stage::getAllGlobalVariables() const { return allGlobalVariables_; }

void Stage::addDoMethod(DoMethodSmartPtr_t& doMethod) {
  DAWN_ASSERT_MSG(
      std::find_if(childrenBegin(), childrenEnd(), [&](const DoMethodSmartPtr_t& doMethodPtr) {
        return doMethodPtr->getInterval() == doMethod->getInterval();
      }) == childrenEnd(), "Do-Method with given interval already exists!");
  insertChild(std::move(doMethod));
  update();
}

void Stage::appendDoMethod(DoMethodSmartPtr_t& from, DoMethodSmartPtr_t& to,
                           const std::shared_ptr<DependencyGraphAccesses>& dependencyGraph) {
  DAWN_ASSERT_MSG(std::find(childrenBegin(), childrenEnd(), to) != childrenEnd(),
                  "'to' DoMethod does not exists");
  DAWN_ASSERT_MSG(from->getInterval() == to->getInterval(),
                  "DoMethods have incompatible intervals!");

  to->setDependencyGraph(dependencyGraph);
  to->insertChildren(to->childrenEnd(), std::make_move_iterator(from->childrenBegin()),
                     std::make_move_iterator(from->childrenEnd()));
  update();
}

LoopOrderKind Stage::getLoopOrder() const { return multiStage_->getLoopOrder(); }

std::vector<std::shared_ptr<Stage>>
Stage::split(std::deque<int>& splitterIndices,
             const std::deque<std::shared_ptr<DependencyGraphAccesses>>* graphs) {
  DAWN_ASSERT_MSG(hasSingleDoMethod(), "Stage::split does not support multiple Do-Methods");
  DoMethod& thisDoMethod = getSingleDoMethod();

  DAWN_ASSERT(thisDoMethod.getChildren().size() >= 2);
  DAWN_ASSERT(!graphs || splitterIndices.size() == graphs->size() - 1);

  std::vector<std::shared_ptr<Stage>> newStages;

  splitterIndices.push_back(thisDoMethod.getChildren().size() - 1);
  DoMethod::StatementAccessesIterator prevSplitterIndex = thisDoMethod.childrenBegin();

  // Create new stages
  for(std::size_t i = 0; i < splitterIndices.size(); ++i) {
    DoMethod::StatementAccessesIterator nextSplitterIndex =
        std::next(thisDoMethod.childrenBegin(), splitterIndices[i] + 1);

    newStages.push_back(std::make_shared<Stage>(stencilInstantiation_, multiStage_,
                                                stencilInstantiation_.nextUID(),
                                                thisDoMethod.getInterval()));
    Stage& newStage = *newStages.back();
    DoMethod& doMethod = newStage.getSingleDoMethod();

    if(graphs)
      doMethod.setDependencyGraph((*graphs)[i]);

    // The new stage contains the statements in the range [prevSplitterIndex , nextSplitterIndex)
    for(auto it = prevSplitterIndex; it != nextSplitterIndex; ++it) {
      doMethod.insertChild(std::move(*it));
    }

    //    for(std::size_t idx = prevSplitterIndex; idx < nextSplitterIndex; ++idx)

    // Update the fields of the new stage
    newStage.update();

    prevSplitterIndex = nextSplitterIndex;
  }

  return newStages;
}

bool Stage::isEmptyOrNullStmt() const {
  for(auto const& doMethod : getChildren()) {
    if(!doMethod->isEmptyOrNullStmt()) {
      return false;
    }
  }
  return true;
}

} // namespace iir
} // namespace dawn
