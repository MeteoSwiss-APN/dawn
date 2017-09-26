//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/Optimizer/Stage.h"
#include "gsl/Optimizer/DependencyGraphAccesses.h"
#include "gsl/Optimizer/StencilInstantiation.h"
#include "gsl/Support/Logging.h"
#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>

namespace gsl {

Stage::Stage(StencilInstantiation* context, MultiStage* multiStage, int StageID,
             const Interval& interval)
    : stencilInstantiation_(context), multiStage_(multiStage), StageID_(StageID) {
  DoMethods_.emplace_back(make_unique<DoMethod>(this, interval));
}

std::vector<std::unique_ptr<DoMethod>>& Stage::getDoMethods() { return DoMethods_; }

const std::vector<std::unique_ptr<DoMethod>>& Stage::getDoMethods() const { return DoMethods_; }

bool Stage::hasSingleDoMethod() const { return (DoMethods_.size() == 1); }

DoMethod& Stage::getSingleDoMethod() {
  GSL_ASSERT_MSG(hasSingleDoMethod(), "stage contains multiple Do-Methods");
  return *DoMethods_.front();
}

const DoMethod& Stage::getSingleDoMethod() const {
  GSL_ASSERT_MSG(hasSingleDoMethod(), "stage contains multiple Do-Methods");
  return *DoMethods_.front();
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
                [&](const Field& field) { verticalExtent.merge(field.Extent[2]); });
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
        if(thisField.AccessID != field.AccessID)
          continue;

        if(thisInterval.extendInterval(thisField.Extent[2])
               .overlaps(interval.extendInterval(field.Extent[2])))
          return true;
      }
    }
  }

  return false;
}

void Stage::update() {
  fields_.clear();
  globalVariables_.clear();

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
  std::set<int> inputOutputFields;
  std::set<int> inputFields;
  std::set<int> outputFields;

  for(const auto& doMethodPtr : DoMethods_) {
    const DoMethod& doMethod = *doMethodPtr;
    for(const auto& statementAccessesPair : doMethod.getStatementAccessesPairs()) {
      const auto& access = statementAccessesPair->getAccesses();

      for(const auto& accessPair : access->getWriteAccesses()) {
        int AccessID = accessPair.first;

        // Does this AccessID correspond to a field access?
        if(!stencilInstantiation_->isField(AccessID)) {
          if(stencilInstantiation_->isGlobalVariable(AccessID))
            globalVariables_.insert(AccessID);
          continue;
        }

        // Field was recorded as `InputOutput`, state can't change ...
        if(inputOutputFields.count(AccessID))
          continue;

        // Field was recorded as `Input`, change it's state to `InputOutput`
        if(inputFields.count(AccessID)) {
          inputOutputFields.insert(AccessID);
          inputFields.erase(AccessID);
          continue;
        }

        // Field not yet present, record it as output
        outputFields.insert(AccessID);
      }

      for(const auto& accessPair : access->getReadAccesses()) {
        int AccessID = accessPair.first;

        // Does this AccessID correspond to a field access?
        if(!stencilInstantiation_->isField(AccessID)) {
          if(stencilInstantiation_->isGlobalVariable(AccessID))
            globalVariables_.insert(AccessID);
          continue;
        }

        // Field was recorded as `InputOutput`, state can't change ...
        if(inputOutputFields.count(AccessID))
          continue;

        // Field was recorded as `Output`, change it's state to `InputOutput`
        if(outputFields.count(AccessID)) {
          inputOutputFields.insert(AccessID);
          outputFields.erase(AccessID);
          continue;
        }

        // Field not yet present, record it as input
        inputFields.insert(AccessID);
      }
    }
  }

  // Merge inputFields, outputFields and fields
  for(int AccessID : outputFields)
    fields_.emplace_back(AccessID, Field::IK_Output);

  for(int AccessID : inputOutputFields)
    fields_.emplace_back(AccessID, Field::IK_InputOutput);

  for(int AccessID : inputFields)
    fields_.emplace_back(AccessID, Field::IK_Input);

  if(fields_.empty()) {
    GSL_LOG(WARNING) << "no fields referenced in stage";
    return;
  }

  // Index to speedup lookup into fields map
  std::unordered_map<int, std::vector<Field>::iterator> AccessIDToFieldMap;
  for(auto it = fields_.begin(), end = fields_.end(); it != end; ++it)
    AccessIDToFieldMap.insert(std::make_pair(it->AccessID, it));

  // Accumulate the extents of each field in this stage
  for(const auto& doMethodPtr : DoMethods_) {
    const DoMethod& doMethod = *doMethodPtr;

    for(const auto& statementAccessesPair : doMethod.getStatementAccessesPairs()) {
      const auto& access = statementAccessesPair->getAccesses();

      // first => AccessID, second => Extent
      for(auto& accessPair : access->getWriteAccesses()) {
        if(!stencilInstantiation_->isField(accessPair.first))
          continue;

        AccessIDToFieldMap[accessPair.first]->Extent.merge(accessPair.second);
      }

      for(const auto& accessPair : access->getReadAccesses()) {
        if(!stencilInstantiation_->isField(accessPair.first))
          continue;

        AccessIDToFieldMap[accessPair.first]->Extent.merge(accessPair.second);
      }
    }
  }
}

const std::unordered_set<int>& Stage::getGlobalVariables() const { return globalVariables_; }

void Stage::addDoMethod(std::unique_ptr<DoMethod>& doMethod) {
  GSL_ASSERT_MSG(std::find_if(DoMethods_.begin(), DoMethods_.end(),
                              [&](const std::unique_ptr<DoMethod>& doMethodPtr) {
                                return doMethodPtr->getInterval() == doMethod->getInterval();
                              }) == DoMethods_.end(),
                 "Do-Method with given interval already exists!");
  DoMethods_.emplace_back(std::move(doMethod));
  update();
}

void Stage::appendDoMethod(std::unique_ptr<DoMethod>& from, std::unique_ptr<DoMethod>& to,
                           const std::shared_ptr<DependencyGraphAccesses>& dependencyGraph) {
  GSL_ASSERT_MSG(std::find(DoMethods_.begin(), DoMethods_.end(), to) != DoMethods_.end(),
                 "'to' DoMethod does not exists");
  GSL_ASSERT_MSG(from->getInterval() == to->getInterval(),
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
  GSL_ASSERT_MSG(hasSingleDoMethod(), "Stage::split does not support multiple Do-Methods");
  DoMethod& thisDoMethod = getSingleDoMethod();
  auto& thisStatementAccessesPairs = thisDoMethod.getStatementAccessesPairs();

  GSL_ASSERT(thisStatementAccessesPairs.size() >= 2);
  GSL_ASSERT(!graphs || splitterIndices.size() == graphs->size() - 1);

  std::vector<std::shared_ptr<Stage>> newStages;

  splitterIndices.push_back(thisStatementAccessesPairs.size() - 1);
  std::size_t prevSplitterIndex = 0;

  // Create new stages
  for(std::size_t i = 0; i < splitterIndices.size(); ++i) {
    std::size_t nextSplitterIndex = splitterIndices[i] + 1;

    newStages.push_back(std::make_shared<Stage>(stencilInstantiation_, multiStage_,
                                                stencilInstantiation_->nextUID(),
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

} // namespace gsl
