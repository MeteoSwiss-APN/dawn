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

#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/Optimizer/ReadBeforeWriteConflict.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/STLExtras.h"
#include "dawn/Optimizer/MultiInterval.h"
#include "dawn/Optimizer/IntervalAlgorithms.h"

namespace dawn {
namespace iir {

MultiStage::MultiStage(StencilInstantiation& stencilInstantiation, LoopOrderKind loopOrder)
    : stencilInstantiation_(stencilInstantiation), loopOrder_(loopOrder) {}

std::vector<std::shared_ptr<MultiStage>>
MultiStage::split(std::deque<MultiStage::SplitIndex>& splitterIndices,
                  LoopOrderKind lastLoopOrder) {

  std::vector<std::shared_ptr<MultiStage>> newMultiStages;

  int curStageIndex = 0;
  auto curStageIt = stages_.begin();
  std::deque<int> curStageSplitterIndices;

  newMultiStages.push_back(std::make_shared<MultiStage>(stencilInstantiation_, lastLoopOrder));

  for(std::size_t i = 0; i < splitterIndices.size(); ++i) {
    SplitIndex& splitIndex = splitterIndices[i];

    if(splitIndex.StageIndex == curStageIndex) {

      curStageSplitterIndices.push_back(splitIndex.StmtIndex);
      newMultiStages.push_back(
          std::make_shared<MultiStage>(stencilInstantiation_, splitIndex.LowerLoopOrder));
      lastLoopOrder = splitIndex.LowerLoopOrder;
    }

    if(i == (splitterIndices.size() - 1) || splitIndex.StageIndex != curStageIndex) {
      if(!curStageSplitterIndices.empty()) {

        // Split the current stage (we assume the graphs are assigned in the stage splitter pass)
        auto newStages = (**curStageIt).split(curStageSplitterIndices, nullptr);

        // Move the new stages to the new MultiStage
        auto newMultiStageRIt = newMultiStages.rbegin();
        for(auto newStagesRIt = newStages.rbegin(); newStagesRIt != newStages.rend();
            ++newStagesRIt, ++newMultiStageRIt)
          (*newMultiStageRIt)->getStages().push_back(std::move(*newStagesRIt));

        curStageSplitterIndices.clear();
      } else {
        // No split in this stage, just move it to the current multi-stage
        newMultiStages.back()->getStages().push_back(std::move(*curStageIt));
      }

      if(i != (splitterIndices.size() - 1))
        newMultiStages.push_back(
            std::make_shared<MultiStage>(stencilInstantiation_, lastLoopOrder));

      // Handle the next stage
      curStageIndex++;
      curStageIt++;
    }
  }

  return newMultiStages;
}

std::shared_ptr<DependencyGraphAccesses>
MultiStage::getDependencyGraphOfInterval(const Interval& interval) const {
  auto dependencyGraph = std::make_shared<DependencyGraphAccesses>(&stencilInstantiation_);
  std::for_each(stages_.begin(), stages_.end(), [&](const std::shared_ptr<Stage>& stagePtr) {
    if(interval.overlaps(stagePtr->getEnclosingExtendedInterval()))
      std::for_each(stagePtr->getDoMethods().begin(), stagePtr->getDoMethods().end(),
                    [&](const std::unique_ptr<DoMethod>& DoMethodPtr) {
                      dependencyGraph->merge(DoMethodPtr->getDependencyGraph().get());
                    });
  });
  return dependencyGraph;
}

std::shared_ptr<DependencyGraphAccesses> MultiStage::getDependencyGraphOfAxis() const {
  auto dependencyGraph = std::make_shared<DependencyGraphAccesses>(&stencilInstantiation_);
  std::for_each(stages_.begin(), stages_.end(), [&](const std::shared_ptr<Stage>& stagePtr) {
    std::for_each(stagePtr->getDoMethods().begin(), stagePtr->getDoMethods().end(),
                  [&](const std::unique_ptr<DoMethod>& DoMethodPtr) {
                    dependencyGraph->merge(DoMethodPtr->getDependencyGraph().get());
                  });
  });
  return dependencyGraph;
}

Cache& MultiStage::setCache(Cache::CacheTypeKind type, Cache::CacheIOPolicy policy, int AccessID,
                            const Interval& interval, boost::optional<Cache::window> w) {
  return caches_.emplace(AccessID,
                         Cache(type, policy, AccessID, boost::optional<Interval>(interval), w))
      .first->second;
}

Cache& MultiStage::setCache(Cache::CacheTypeKind type, Cache::CacheIOPolicy policy, int AccessID) {
  return caches_.emplace(AccessID, Cache(type, policy, AccessID, boost::optional<Interval>(),
                                         boost::optional<Cache::window>()))
      .first->second;
}

std::vector<DoMethod> MultiStage::computeOrderedDoMethods() const {
  auto intervals_set = getIntervals();
  std::vector<Interval> intervals_v;
  std::copy(intervals_set.begin(), intervals_set.end(), std::back_inserter(intervals_v));

  // compute the partition of the intervals
  auto partitionIntervals = Interval::computePartition(intervals_v);
  if((getLoopOrder() == LoopOrderKind::LK_Backward))
    std::reverse(partitionIntervals.begin(), partitionIntervals.end());

  std::vector<DoMethod> orderedDoMethods;

  for(auto interval : partitionIntervals) {
    for(const auto& stagePtr : getStages()) {
      for(const auto& doMethod : stagePtr->getDoMethods()) {

        if(doMethod->getInterval().overlaps(interval)) {
          DoMethod partitionedDoMethod(*doMethod);

          partitionedDoMethod.setInterval(interval);
          orderedDoMethods.push_back(partitionedDoMethod);
          // there should not be two do methods in the same stage with overlapping intervals
          continue;
        }
      }
    }
  }

  return orderedDoMethods;
}

MultiInterval MultiStage::computeReadAccessInterval(int accessID) const {

  std::vector<DoMethod> orderedDoMethods = computeOrderedDoMethods();

  MultiInterval writeInterval;
  MultiInterval writeIntervalPre;
  MultiInterval readInterval;

  for(const auto& doMethod : orderedDoMethods) {
    for(const auto& statementAccesssPair : doMethod.getStatementAccessesPairs()) {
      const Accesses& accesses = *statementAccesssPair->getAccesses();
      if(accesses.hasWriteAccess(accessID)) {
        writeIntervalPre.insert(doMethod.getInterval());
      }
    }
  }

  for(const auto& doMethod : orderedDoMethods) {
    for(const auto& statementAccesssPair : doMethod.getStatementAccessesPairs()) {
      const Accesses& accesses = *statementAccesssPair->getAccesses();
      // indepdently of whether the statement has also a write access, if there is a read
      // access, it should happen in the RHS so first
      if(accesses.hasReadAccess(accessID)) {
        MultiInterval interv;

        Extents const& readAccessExtent = accesses.getReadAccess(accessID);
        boost::optional<Extent> readAccessInLoopOrder = readAccessExtent.getVerticalLoopOrderExtent(
            getLoopOrder(), Extents::VerticalLoopOrderDir::VL_InLoopOrder, false);
        Interval computingInterval = doMethod.getInterval();
        if(readAccessInLoopOrder.is_initialized()) {
          interv.insert(computingInterval.extendInterval(*readAccessInLoopOrder));
        }
        if(!writeIntervalPre.empty()) {
          interv.substract(writeIntervalPre);
        }

        if(readAccessExtent.hasVerticalCenter()) {
          auto centerAccessInterval = substract(computingInterval, writeInterval);
          interv.insert(centerAccessInterval);
        }

        boost::optional<Extent> readAccessCounterLoopOrder =
            readAccessExtent.getVerticalLoopOrderExtent(
                getLoopOrder(), Extents::VerticalLoopOrderDir::VL_CounterLoopOrder, false);

        if(readAccessCounterLoopOrder.is_initialized()) {
          interv.insert(computingInterval.extendInterval(*readAccessCounterLoopOrder));
        }

        readInterval.insert(interv);
      }
      if(accesses.hasWriteAccess(accessID)) {
        writeInterval.insert(doMethod.getInterval());
      }
    }
  }

  return readInterval;
}

boost::optional<Interval>
MultiStage::computeEnclosingAccessInterval(const int accessID,
                                           const bool mergeWithDoInterval) const {
  boost::optional<Interval> interval;
  for(auto const& stage : stages_) {
    boost::optional<Interval> doInterval =
        stage->computeEnclosingAccessInterval(accessID, mergeWithDoInterval);

    if(doInterval) {
      if(interval)
        (*interval).merge(*doInterval);
      else
        interval = doInterval;
    }
  }
  return interval;
}

std::unordered_set<Interval> MultiStage::getIntervals() const {
  std::unordered_set<Interval> intervals;
  for(const auto& stagePtr : stages_)
    for(const auto& doMethodPtr : stagePtr->getDoMethods())
      intervals.insert(doMethodPtr->getInterval());
  return intervals;
}

Interval MultiStage::getEnclosingInterval() const {
  DAWN_ASSERT(!stages_.empty());
  Interval interval = (*stages_.begin())->getEnclosingInterval();

  for(auto it = std::next(stages_.begin()), end = stages_.end(); it != end; ++it)
    interval.merge((*stages_.begin())->getEnclosingInterval());

  return interval;
}

boost::optional<Interval> MultiStage::getEnclosingAccessIntervalTemporaries() const {
  boost::optional<Interval> interval;
  // notice we dont use here the fields of getFields() since they contain the enclosing of all the
  // extents and intervals of all stages and it would give larger intervals than really required
  // inspecting the extents and intervals of individual stages
  for(const auto& stagePtr : stages_) {
    for(const Field& field : stagePtr->getFields()) {
      int AccessID = field.getAccessID();
      if(!stencilInstantiation_.isTemporaryField(AccessID))
        continue;

      if(!interval.is_initialized()) {
        interval = boost::make_optional(field.computeAccessedInterval());
      } else {
        interval->merge(field.computeAccessedInterval());
      }
    }
  }

  return interval;
}

std::unordered_map<int, Field> MultiStage::getFields() const {
  std::unordered_map<int, Field> fields;

  for(const auto& stagePtr : stages_) {
    for(const Field& field : stagePtr->getFields()) {
      auto it = fields.find(field.getAccessID());
      if(it != fields.end()) {
        // Adjust the Intend
        if(it->second.getIntend() == Field::IK_Input && field.getIntend() == Field::IK_Output)
          it->second.setIntend(Field::IK_InputOutput);
        else if(it->second.getIntend() == Field::IK_Output && field.getIntend() == Field::IK_Input)
          it->second.setIntend(Field::IK_InputOutput);

        // Merge the Extent
        it->second.mergeReadExtents(field.getReadExtents());
        it->second.mergeWriteExtents(field.getWriteExtents());
        it->second.extendInterval(field.getInterval());
      } else
        fields.emplace(field.getAccessID(), field);
    }
  }

  return fields;
}

void MultiStage::renameAllOccurrences(int oldAccessID, int newAccessID) {
  for(auto stageIt = getStages().begin(); stageIt != getStages().end(); ++stageIt) {
    Stage& stage = (**stageIt);
    for(auto& doMethodPtr : stage.getDoMethods()) {
      DoMethod& doMethod = *doMethodPtr;
      renameAccessIDInStmts(&stencilInstantiation_, oldAccessID, newAccessID,
                            doMethod.getStatementAccessesPairs());
      renameAccessIDInAccesses(&stencilInstantiation_, oldAccessID, newAccessID,
                               doMethod.getStatementAccessesPairs());
    }

    stage.update();
  }
}

bool MultiStage::isEmptyOrNullStmt() const {
  for(auto stageIt = getStages().begin(); stageIt != getStages().end(); ++stageIt) {
    if(!(*stageIt)->isEmptyOrNullStmt())
      return false;
  }
  return true;
}

} // namespace iir
} // namespace dawn
