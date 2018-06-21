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

#include "dawn/Optimizer/Stencil.h"
#include "dawn/Optimizer/DependencyGraphStage.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace dawn {

std::ostream& operator<<(std::ostream& os, const Stencil::StagePosition& position) {
  return (os << "(" << position.MultiStageIndex << ", " << position.StageOffset << ")");
}

std::ostream& operator<<(std::ostream& os, const Stencil::StatementPosition& position) {
  return (os << "(Stage=" << position.StagePos << ", DoMethod=" << position.DoMethodIndex
             << ", Statement=" << position.StatementIndex << ")");
}

std::ostream& operator<<(std::ostream& os, const Stencil::Lifetime& lifetime) {
  return (os << "[Begin=" << lifetime.Begin << ", End=" << lifetime.End << "]");
}

bool Stencil::StagePosition::operator<(const Stencil::StagePosition& other) const {
  return MultiStageIndex < other.MultiStageIndex ||
         (MultiStageIndex == other.MultiStageIndex && StageOffset < other.StageOffset);
}

bool Stencil::StagePosition::operator==(const Stencil::StagePosition& other) const {
  return MultiStageIndex == other.MultiStageIndex && StageOffset == other.StageOffset;
}

bool Stencil::StagePosition::operator!=(const Stencil::StagePosition& other) const {
  return !(*this == other);
}

bool Stencil::StatementPosition::operator<(const Stencil::StatementPosition& other) const {
  return StagePos < other.StagePos ||
         (StagePos == other.StagePos && DoMethodIndex == other.DoMethodIndex &&
          StatementIndex < other.StatementIndex);
}

bool Stencil::StatementPosition::operator<=(const Stencil::StatementPosition& other) const {
  return operator<(other) || operator==(other);
}

bool Stencil::StatementPosition::operator==(const Stencil::StatementPosition& other) const {
  return StagePos == other.StagePos && DoMethodIndex == other.DoMethodIndex &&
         StatementIndex == other.StatementIndex;
}

bool Stencil::StatementPosition::operator!=(const Stencil::StatementPosition& other) const {
  return !(*this == other);
}

bool Stencil::StatementPosition::inSameDoMethod(const Stencil::StatementPosition& other) const {
  return StagePos == other.StagePos && DoMethodIndex == other.DoMethodIndex;
}

bool Stencil::Lifetime::overlaps(const Stencil::Lifetime& other) const {
  // Note: same stage but different Do-Method are treated as overlapping!

  bool lowerBoundOverlap = false;
  if(Begin.StagePos == other.End.StagePos && Begin.DoMethodIndex != other.End.DoMethodIndex)
    lowerBoundOverlap = true;
  else
    lowerBoundOverlap = Begin <= other.End;

  bool upperBoundOverlap = false;
  if(other.Begin.StagePos == End.StagePos && other.Begin.DoMethodIndex != End.DoMethodIndex)
    upperBoundOverlap = true;
  else
    upperBoundOverlap = other.Begin <= End;

  return lowerBoundOverlap && upperBoundOverlap;
}

Stencil::Stencil(StencilInstantiation& stencilInstantiation,
                 const std::shared_ptr<sir::Stencil>& SIRStencil, int StencilID,
                 const std::shared_ptr<DependencyGraphStage>& stageDependencyGraph)
    : stencilInstantiation_(stencilInstantiation), SIRStencil_(SIRStencil), StencilID_(StencilID),
      stageDependencyGraph_(stageDependencyGraph) {}

std::unordered_set<Interval> Stencil::getIntervals() const {
  std::unordered_set<Interval> intervals;
  for(const auto& multistage : multistages_)
    for(const auto& stage : multistage->getStages())
      for(const auto& doMethod : stage->getDoMethods())
        intervals.insert(doMethod->getInterval());

  return intervals;
}

std::vector<Stencil::FieldInfo> Stencil::getFields(bool withTemporaries) const {
  std::set<int> fieldAccessIDs;
  for(const auto& multistage : multistages_) {
    for(const auto& stage : multistage->getStages()) {
      for(const auto& field : stage->getFields()) {
        fieldAccessIDs.insert(field.getAccessID());
      }
    }
  }

  std::vector<FieldInfo> fields;

  for(const auto& AccessID : fieldAccessIDs) {
    std::string name = stencilInstantiation_.getNameFromAccessID(AccessID);
    bool isTemporary = stencilInstantiation_.isTemporaryField(AccessID);

    if(isTemporary) {
      if(withTemporaries) {
        fields.insert(fields.begin(), FieldInfo{isTemporary, name, AccessID});
      }
    } else {
      fields.emplace_back(FieldInfo{isTemporary, name, AccessID});
    }
  }

  return fields;
}

std::vector<std::string> Stencil::getGlobalVariables() const {
  std::set<int> globalVariableAccessIDs;
  for(const auto& multistage : multistages_) {
    for(const auto& stage : multistage->getStages()) {
      globalVariableAccessIDs.insert(stage->getAllGlobalVariables().begin(),
                                     stage->getAllGlobalVariables().end());
    }
  }

  std::vector<std::string> globalVariables;
  for(const auto& AccessID : globalVariableAccessIDs)
    globalVariables.push_back(stencilInstantiation_.getNameFromAccessID(AccessID));

  return globalVariables;
}

int Stencil::getNumStages() const {
  return std::accumulate(multistages_.begin(), multistages_.end(), int(0),
                         [](int numStages, const std::shared_ptr<MultiStage>& MS) {
                           return numStages + MS->getStages().size();
                         });
}

void Stencil::forEachStatementAccessesPair(
    std::function<void(ArrayRef<std::shared_ptr<StatementAccessesPair>>)> func, bool updateFields) {
  forEachStatementAccessesPairImpl(func, 0, getNumStages(), updateFields);
}

void Stencil::forEachStatementAccessesPair(
    std::function<void(ArrayRef<std::shared_ptr<StatementAccessesPair>>)> func,
    const Stencil::Lifetime& lifetime, bool updateFields) {
  int startStageIdx = getStageIndexFromPosition(lifetime.Begin.StagePos);
  int endStageIdx = getStageIndexFromPosition(lifetime.End.StagePos);
  forEachStatementAccessesPairImpl(func, startStageIdx, endStageIdx + 1, updateFields);
}

void Stencil::forEachStatementAccessesPairImpl(
    std::function<void(ArrayRef<std::shared_ptr<StatementAccessesPair>>)> func, int startStageIdx,
    int endStageIdx, bool updateFields) {
  for(int stageIdx = startStageIdx; stageIdx < endStageIdx; ++stageIdx) {
    auto stage = getStage(stageIdx);
    for(const auto& doMethodPtr : stage->getDoMethods())
      func(doMethodPtr->getStatementAccessesPairs());

    if(updateFields)
      stage->update();
  }
}

void Stencil::updateFields(const Stencil::Lifetime& lifetime) {
  int startStageIdx = getStageIndexFromPosition(lifetime.Begin.StagePos);
  int endStageIdx = getStageIndexFromPosition(lifetime.End.StagePos);
  updateFieldsImpl(startStageIdx, endStageIdx + 1);
}

void Stencil::updateFields() { updateFieldsImpl(0, getNumStages()); }

void Stencil::updateFieldsImpl(int startStageIdx, int endStageIdx) {
  for(int stageIdx = startStageIdx; stageIdx < endStageIdx; ++stageIdx)
    getStage(stageIdx)->update();
}

void Stencil::setStageDependencyGraph(const std::shared_ptr<DependencyGraphStage>& stageDAG) {
  stageDependencyGraph_ = stageDAG;
}

const std::shared_ptr<DependencyGraphStage>& Stencil::getStageDependencyGraph() const {
  return stageDependencyGraph_;
}

const std::shared_ptr<MultiStage>&
Stencil::getMultiStageFromMultiStageIndex(int multiStageIdx) const {
  DAWN_ASSERT_MSG(multiStageIdx < multistages_.size(), "invalid multi-stage index");
  auto msIt = multistages_.begin();
  std::advance(msIt, multiStageIdx);
  return *msIt;
}

const std::shared_ptr<MultiStage>& Stencil::getMultiStageFromStageIndex(int stageIdx) const {
  return getMultiStageFromMultiStageIndex(getPositionFromStageIndex(stageIdx).MultiStageIndex);
}

Stencil::StagePosition Stencil::getPositionFromStageIndex(int stageIdx) const {
  DAWN_ASSERT(!multistages_.empty());
  if(stageIdx == -1)
    return StagePosition(0, -1);

  int curIdx = 0, multiStageIdx = 0;
  for(const auto& MS : multistages_) {

    // Is our stage in this multi-stage?
    int numStages = MS->getStages().size();
    if((curIdx + numStages) <= stageIdx) {
      curIdx += numStages;
      multiStageIdx++;
      continue;
    } else {
      int stageOffset = stageIdx - curIdx;
      DAWN_ASSERT_MSG(stageOffset < numStages, "invalid stage index");
      return StagePosition(multiStageIdx, stageOffset);
    }
  }
  dawn_unreachable("invalid stage index");
}

int Stencil::getStageIndexFromPosition(const Stencil::StagePosition& position) const {
  auto curMSIt = multistages_.begin();
  std::advance(curMSIt, position.MultiStageIndex);

  // Count the number of stages in the multistages before our current MS
  int numStagesInMSBeforeCurMS =
      std::accumulate(multistages_.begin(), curMSIt, int(0),
                      [&](int numStages, const std::shared_ptr<MultiStage>& MS) {
                        return numStages + MS->getStages().size();
                      });

  // Add the current stage offset
  return numStagesInMSBeforeCurMS + position.StageOffset;
}

const std::shared_ptr<Stage>& Stencil::getStage(const StagePosition& position) const {
  // Get the multi-stage ...
  DAWN_ASSERT_MSG(position.MultiStageIndex < multistages_.size(), "invalid multi-stage index");
  auto msIt = multistages_.begin();
  std::advance(msIt, position.MultiStageIndex);
  const auto& MS = *msIt;

  // ... and the requested stage inside the given multi-stage
  DAWN_ASSERT_MSG(position.StageOffset == -1 || position.StageOffset < MS->getStages().size(),
                  "invalid stage offset");
  auto stageIt = MS->getStages().begin();
  std::advance(stageIt, position.StageOffset == -1 ? 0 : position.StageOffset);
  return *stageIt;
}

const std::shared_ptr<Stage>& Stencil::getStage(int stageIdx) const {
  int curIdx = 0;
  for(const auto& MS : multistages_) {

    // Is our stage in this multi-stage?
    int numStages = MS->getStages().size();

    if((curIdx + numStages) <= stageIdx) {
      // No... continue
      curIdx += numStages;
      continue;
    } else {
      // Yes... advance to our stage
      int stageOffset = stageIdx - curIdx;

      DAWN_ASSERT_MSG(stageOffset < MS->getStages().size(), "invalid stage index");
      auto stageIt = MS->getStages().begin();
      std::advance(stageIt, stageOffset);

      return *stageIt;
    }
  }
  dawn_unreachable("invalid stage index");
}

void Stencil::insertStage(const StagePosition& position, const std::shared_ptr<Stage>& stage) {

  // Get the multi-stage ...
  DAWN_ASSERT_MSG(position.MultiStageIndex < multistages_.size(), "invalid multi-stage index");
  auto msIt = multistages_.begin();
  std::advance(msIt, position.MultiStageIndex);
  const auto& MS = *msIt;

  // ... and the requested stage inside the given multi-stage
  DAWN_ASSERT_MSG(position.StageOffset == -1 || position.StageOffset < MS->getStages().size(),
                  "invalid stage offset");
  auto stageIt = MS->getStages().begin();

  // A stage offset of -1 indicates *before* the first element (thus nothing to do).
  // Otherwise we advance one beyond the requested stage as we insert *after* the specified
  // stage and `std::list::insert` inserts *before*.
  if(position.StageOffset != -1) {
    std::advance(stageIt, position.StageOffset);
    if(stageIt != MS->getStages().end())
      stageIt++;
  }

  MS->getStages().insert(stageIt, stage);
}

Interval Stencil::getAxis(bool useExtendedInterval) const {
  int numStages = getNumStages();
  DAWN_ASSERT_MSG(numStages, "need atleast one stage");

  Interval axis = getStage(0)->getEnclosingExtendedInterval();
  for(int stageIdx = 1; stageIdx < numStages; ++stageIdx)
    axis.merge(useExtendedInterval ? getStage(stageIdx)->getEnclosingExtendedInterval()
                                   : getStage(stageIdx)->getEnclosingInterval());
  return axis;
}

void Stencil::renameAllOccurrences(int oldAccessID, int newAccessID) {
  for(const auto& multistage : getMultiStages()) {
    multistage->renameAllOccurrences(oldAccessID, newAccessID);
  }
}

std::unordered_map<int, Stencil::Lifetime>
Stencil::getLifetime(const std::unordered_set<int>& AccessIDs) const {
  std::unordered_map<int, StatementPosition> Begin;
  std::unordered_map<int, StatementPosition> End;

  int multiStageIdx = 0;
  for(const auto& multistagePtr : multistages_) {

    int stageOffset = 0;
    for(const auto& stagePtr : multistagePtr->getStages()) {

      int doMethodIndex = 0;
      for(const auto& doMethodPtr : stagePtr->getDoMethods()) {
        DoMethod& doMethod = *doMethodPtr;

        for(int statementIdx = 0; statementIdx < doMethod.getStatementAccessesPairs().size();
            ++statementIdx) {
          const Accesses& accesses =
              *doMethod.getStatementAccessesPairs()[statementIdx]->getAccesses();

          auto processAccessMap = [&](const std::unordered_map<int, Extents>& accessMap) {
            for(const auto& AccessIDExtentPair : accessMap) {
              int AccessID = AccessIDExtentPair.first;

              if(AccessIDs.count(AccessID)) {
                StatementPosition pos(StagePosition(multiStageIdx, stageOffset), doMethodIndex,
                                      statementIdx);

                if(!Begin.count(AccessID))
                  Begin.emplace(AccessID, pos);
                End[AccessID] = pos;
              }
            }
          };

          processAccessMap(accesses.getWriteAccesses());
          processAccessMap(accesses.getReadAccesses());
        }

        doMethodIndex++;
      }

      stageOffset++;
    }

    multiStageIdx++;
  }

  std::unordered_map<int, Lifetime> lifetimeMap;
  for(int AccessID : AccessIDs) {
    auto& begin = Begin[AccessID];
    auto& end = End[AccessID];

    lifetimeMap.emplace(AccessID, Lifetime(begin, end));
  }

  return lifetimeMap;
}

bool Stencil::isEmpty() const {
  for(const auto& MS : getMultiStages())
    for(const auto& stage : MS->getStages())
      for(auto& doMethod : stage->getDoMethods())
        if(!doMethod->getStatementAccessesPairs().empty())
          return false;
  return true;
}

std::shared_ptr<Interval> Stencil::getEnclosingIntervalTemporaries() const {
  std::shared_ptr<Interval> tmpInterval;
  for(auto mss : getMultiStages()) {
    auto mssInterval = mss->getEnclosingAccessIntervalTemporaries();
    if(tmpInterval != nullptr && mssInterval != nullptr) {
      tmpInterval->merge(*mssInterval);
    } else if(mssInterval != nullptr) {
      tmpInterval = mssInterval;
    }
  }
  return tmpInterval;
}

const std::shared_ptr<sir::Stencil> Stencil::getSIRStencil() const { return SIRStencil_; }

void Stencil::accept(ASTVisitor& visitor) {
  for(const auto& multistagePtr : multistages_)
    for(const auto& stagePtr : multistagePtr->getStages())
      for(const auto& doMethodPtr : stagePtr->getDoMethods())
        for(const auto& stmtAcessesPairPtr : doMethodPtr->getStatementAccessesPairs())
          stmtAcessesPairPtr->getStatement()->ASTStmt->accept(visitor);
}

std::ostream& operator<<(std::ostream& os, const Stencil& stencil) {
  int multiStageIdx = 0;
  for(const auto& MS : stencil.getMultiStages()) {
    os << "MultiStage " << (multiStageIdx++) << ": (" << MS->getLoopOrder() << ")\n";
    for(const auto& stage : MS->getStages())
      os << "  " << stencil.getStencilInstantiation().getNameFromStageID(stage->getStageID()) << " "
         << RangeToString()(stage->getFields(),
                            [&](const Field& field) {
                              return stencil.getStencilInstantiation().getNameFromAccessID(
                                  field.getAccessID());
                            })
         << "\n";
  }
  return os;
}

std::unordered_map<int, Extents> const Stencil::computeEnclosingAccessExtents() const {
  std::unordered_map<int, Extents> maxExtents_;
  // iterate through multistages
  for(const auto& MS : multistages_) {
    // iterate through stages
    for(const auto& stage : MS->getStages()) {
      std::size_t accessorIdx = 0;
      for(; accessorIdx < stage->getFields().size(); ++accessorIdx) {
        const auto& field = stage->getFields()[accessorIdx];
        // add the stage extent to the field extent
        Extents e = field.getExtents();
        e.add(stage->getExtents());
        // merge with the current minimum/maximum extent for the given field
        maxExtents_[stage->getFields()[accessorIdx].getAccessID()].merge(e);
      }
    }
  }
  return maxExtents_;
}

} // namespace dawn
