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
#include "dawn/AST/ASTStmt.h"
#include "dawn/AST/LocationType.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/Support/Logger.h"
#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>

namespace dawn {
namespace iir {

Stage::Stage(const StencilMetaInformation& metaData, int StageID, IterationSpace iterationSpace)
    : metaData_(metaData), StageID_(StageID), iterationSpace_(iterationSpace) {}

Stage::Stage(const StencilMetaInformation& metaData, int StageID, const Interval& interval,
             IterationSpace iterationSpace)
    : metaData_(metaData), StageID_(StageID), iterationSpace_(iterationSpace) {
  insertChild(std::make_unique<DoMethod>(interval, metaData));
}

json::json Stage::jsonDump(const StencilMetaInformation& metaData) const {
  json::json node;
  json::json fieldsJson;
  for(const auto& field : derivedInfo_.fields_) {
    fieldsJson[metaData.getNameFromAccessID(field.first)] = field.second.jsonDump();
  }
  node["Fields"] = fieldsJson;
  std::stringstream ss;
  ss << derivedInfo_.extents_;
  node["Extents"] = ss.str();
  node["RequiresSync"] = derivedInfo_.requiresSync_;

  auto doMethodsArray = json::json::array();
  for(const auto& doMethod : children_) {
    doMethodsArray.push_back(doMethod->jsonDump(metaData));
  }
  node["DoMethods"] = doMethodsArray;
  return node;
}

std::unique_ptr<Stage> Stage::clone() const {

  auto cloneStage = std::make_unique<Stage>(metaData_, StageID_);

  cloneStage->derivedInfo_ = derivedInfo_;
  cloneStage->type_ = type_;

  cloneStage->cloneChildrenFrom(*this);
  return cloneStage;
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

void Stage::setRequiresSync(const bool sync) { derivedInfo_.requiresSync_ = sync; }
bool Stage::getRequiresSync() const { return derivedInfo_.requiresSync_; }

std::optional<Interval>
Stage::computeEnclosingAccessInterval(const int accessID, const bool mergeWithDoInterval) const {
  std::optional<Interval> interval;
  for(auto const& doMethod : getChildren()) {
    std::optional<Interval> doInterval =
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
  DAWN_ASSERT_MSG(getChildren().size() > 0, "Stage does not contain any children");
  Interval interval = getChildren().front()->getInterval();
  for(const auto& doMethod : getChildren())
    interval.merge(doMethod->getInterval());
  return interval;
}

Extent Stage::getMaxVerticalExtent() const {
  Extent verticalExtent;
  std::for_each(derivedInfo_.fields_.begin(), derivedInfo_.fields_.end(),
                [&](const std::pair<int, Field>& pair) {
                  verticalExtent.merge(pair.second.getExtents().verticalExtent());
                });
  return verticalExtent;
}

Interval Stage::getEnclosingExtendedInterval() const {
  return getEnclosingInterval().extendInterval(getMaxVerticalExtent());
}

void Stage::DerivedInfo::clear() {
  fields_.clear();
  globalVariables_.clear();
  globalVariablesFromStencilFunctionCalls_.clear();
  allGlobalVariables_.clear();
}

void Stage::clearDerivedInfo() { derivedInfo_.clear(); }
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

        if(thisInterval.extendInterval(thisField.getExtents().verticalExtent())
               .overlaps(interval.extendInterval(field.getExtents().verticalExtent())))
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
class CaptureStencilFunctionCallGlobalParams : public iir::ASTVisitorForwarding {

  std::unordered_set<int>& globalVariables_;
  const StencilMetaInformation& metaData_;
  std::shared_ptr<const StencilFunctionInstantiation> function_;

public:
  CaptureStencilFunctionCallGlobalParams(std::unordered_set<int>& globalVariables,
                                         const StencilMetaInformation& metaData)
      : globalVariables_(globalVariables), metaData_(metaData), function_(nullptr) {}

  void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override {
    // Find the referenced stencil function
    std::shared_ptr<const StencilFunctionInstantiation> stencilFun =
        function_ ? function_->getStencilFunctionInstantiation(expr)
                  : metaData_.getStencilFunctionInstantiation(expr);

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

void Stage::updateLevel() { updateGlobalVariablesInfo(); }

void Stage::updateGlobalVariablesInfo() {
  CaptureStencilFunctionCallGlobalParams functionCallGlobaParamVisitor(
      derivedInfo_.globalVariablesFromStencilFunctionCalls_, metaData_);

  for(const auto& doMethodPtr : getChildren()) {
    const DoMethod& doMethod = *doMethodPtr;
    for(const auto& stmt : doMethod.getAST().getStatements()) {
      const auto& access = stmt->getData<IIRStmtData>().CallerAccesses;
      DAWN_ASSERT(access);
      for(const auto& accessPair : access->getWriteAccesses()) {
        int AccessID = accessPair.first;
        // Does this AccessID correspond to a field access?
        if(metaData_.isAccessType(iir::FieldAccessType::GlobalVariable, AccessID)) {
          derivedInfo_.globalVariables_.insert(AccessID);
        }
      }
      for(const auto& accessPair : access->getReadAccesses()) {
        int AccessID = accessPair.first;
        if(metaData_.isAccessType(iir::FieldAccessType::GlobalVariable, AccessID)) {
          derivedInfo_.globalVariables_.insert(AccessID);
        }
      }

      // capture all the accesses to global accesses of stencil function called
      // from this statement
      stmt->accept(functionCallGlobaParamVisitor);
    }
  }

  derivedInfo_.allGlobalVariables_.insert(derivedInfo_.globalVariables_.begin(),
                                          derivedInfo_.globalVariables_.end());
  derivedInfo_.allGlobalVariables_.insert(
      derivedInfo_.globalVariablesFromStencilFunctionCalls_.begin(),
      derivedInfo_.globalVariablesFromStencilFunctionCalls_.end());
}
bool Stage::hasGlobalVariables() const {
  return (!derivedInfo_.globalVariables_.empty()) ||
         (!derivedInfo_.globalVariablesFromStencilFunctionCalls_.empty());
}

const std::unordered_set<int>& Stage::getGlobalVariables() const {
  return derivedInfo_.globalVariables_;
}

const std::unordered_set<int>& Stage::getGlobalVariablesFromStencilFunctionCalls() const {
  return derivedInfo_.globalVariablesFromStencilFunctionCalls_;
}

const std::unordered_set<int>& Stage::getAllGlobalVariables() const {
  return derivedInfo_.allGlobalVariables_;
}

void Stage::addDoMethod(DoMethodSmartPtr_t&& doMethod) {
  DAWN_ASSERT_MSG(std::find_if(childrenBegin(), childrenEnd(),
                               [&](const DoMethodSmartPtr_t& doMethodPtr) {
                                 return doMethodPtr->getInterval() == doMethod->getInterval();
                               }) == childrenEnd(),
                  "Do-Method with given interval already exists!");

  insertChild(std::move(doMethod));
}

void Stage::appendDoMethod(DoMethodSmartPtr_t& from, DoMethodSmartPtr_t& to,
                           DependencyGraphAccesses&& dependencyGraph) {
  DAWN_ASSERT_MSG(std::find(childrenBegin(), childrenEnd(), to) != childrenEnd(),
                  "'to' DoMethod does not exists");
  DAWN_ASSERT_MSG(from->getInterval() == to->getInterval(),
                  "DoMethods have incompatible intervals!");

  to->setDependencyGraph(std::move(dependencyGraph));
  to->getAST().insert_back(std::make_move_iterator(from->getAST().getStatements().begin()),
                           std::make_move_iterator(from->getAST().getStatements().end()));
}

static std::deque<std::pair<ast::BlockStmt::StmtConstIterator, ast::BlockStmt::StmtConstIterator>>
convertSplitterIndicesToRanges(ast::BlockStmt::StmtConstIterator beginIterator,
                               ast::BlockStmt::StmtConstIterator endIterator,
                               std::deque<int> const& splitterIndices) {
  std::deque<std::pair<ast::BlockStmt::StmtConstIterator, ast::BlockStmt::StmtConstIterator>>
      ranges;
  auto prevIterator = beginIterator;
  for(auto splitterIndex : splitterIndices) {
    auto nextIterator = std::next(beginIterator, splitterIndex + 1);
    ranges.emplace_back(prevIterator, nextIterator);
    prevIterator = nextIterator;
  }
  ranges.emplace_back(prevIterator, endIterator);
  return ranges;
}

std::vector<std::unique_ptr<Stage>> Stage::split(std::deque<int> const& splitterIndices) {
  DAWN_ASSERT_MSG(hasSingleDoMethod(), "Stage::split does not support multiple Do-Methods");
  const DoMethod& thisDoMethod = getSingleDoMethod();

  DAWN_ASSERT(thisDoMethod.getAST().getStatements().size() >= 2);

  auto ranges =
      convertSplitterIndicesToRanges(thisDoMethod.getAST().getStatements().begin(),
                                     thisDoMethod.getAST().getStatements().end(), splitterIndices);

  std::vector<std::unique_ptr<Stage>> newStages;
  for(auto const& [beginIter, endIter] : ranges) {
    newStages.push_back(std::make_unique<Stage>(metaData_, UIDGenerator::getInstance()->get(),
                                                thisDoMethod.getInterval()));
    Stage& newStage = *newStages.back();
    newStage.setIterationSpace(thisDoMethod.getParent()->getIterationSpace());
    DoMethod& doMethod = newStage.getSingleDoMethod();

    doMethod.getAST().insert_back(beginIter, endIter);

    // Update the fields of the new doMethod
    doMethod.update(NodeUpdateType::level);
    newStage.update(NodeUpdateType::level);
  }

  return newStages;
}

std::vector<std::unique_ptr<Stage>> Stage::split(std::deque<int> const& splitterIndices,
                                                 std::deque<DependencyGraphAccesses>&& graphs) {
  DAWN_ASSERT(splitterIndices.size() == graphs.size() - 1);
  auto newStages = split(splitterIndices);
  for(std::size_t i = 0; i < newStages.size(); ++i) {
    DoMethod& doMethod = newStages[i]->getSingleDoMethod();
    doMethod.setDependencyGraph(std::move(graphs[i]));
  }
  return newStages;
}

void Stage::updateFromChildren() {
  updateGlobalVariablesInfo();

  for(const auto& doMethod : children_) {
    mergeFields(doMethod->getFields(), derivedInfo_.fields_, std::optional<Extents>());
  }
}

bool Stage::isEmptyOrNullStmt() const {
  for(auto const& doMethod : getChildren()) {
    if(!doMethod->isEmptyOrNullStmt()) {
      return false;
    }
  }
  return true;
}

void Stage::setLocationType(ast::LocationType type) { type_ = type; }

std::optional<ast::LocationType> Stage::getLocationType() const { return type_; }

void Stage::setIterationSpace(const IterationSpace& value) { iterationSpace_ = value; }

const Stage::IterationSpace& Stage::getIterationSpace() const { return iterationSpace_; }

bool Stage::hasIterationSpace() const {
  return std::any_of(iterationSpace_.cbegin(), iterationSpace_.cend(),
                     [](const auto& p) { return p.has_value(); });
}

bool Stage::iterationSpaceCompatible(const Stage& other) const {
  bool compatible = true;

  if((iterationSpace_[0].has_value() && !other.getIterationSpace()[0].has_value()) ||
     (iterationSpace_[1].has_value() && !other.getIterationSpace()[1].has_value())) {
    return false;
  }

  if(iterationSpace_[0].has_value() && other.getIterationSpace()[0].has_value()) {
    compatible &= iterationSpace_[0]->contains(*other.getIterationSpace()[0]);
  }
  if(iterationSpace_[1].has_value() && other.getIterationSpace()[1].has_value()) {
    compatible &= iterationSpace_[1]->contains(*other.getIterationSpace()[1]);
  }
  return compatible;
}

} // namespace iir
} // namespace dawn
