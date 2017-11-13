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

#include "dawn/Optimizer/PassSetNonTempCaches.h"
#include "dawn/Optimizer/Cache.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassDataLocalityMetric.h"
#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/ASTExpr.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>
#include <set>
#include <vector>

namespace dawn {

struct accessIDaccessMetric {
  accessIDaccessMetric(int id, int gain) : accessID(id), dataLocalityGain(gain) {}
  int accessID;
  int dataLocalityGain;
  bool operator<(const accessIDaccessMetric& rhs) {
    return dataLocalityGain < rhs.dataLocalityGain;
  }
};

class GlobalFieldCacher {
public:
  GlobalFieldCacher(MultiStage* msptr, StencilInstantiation* si)
      : multiStagePrt_(msptr), instantiation_(si) {}

  void process() {
    computeOptimalFields();
    addFillerStages();
    addFlushStages();
  }

  std::vector<accessIDaccessMetric>& getSortedAccesses() { return sortedAccesses_; }
  std::unordered_map<int, Cache> getIDtocacheMap() { return idToCache_; }
  std::unordered_map<int, int> getaccessIDtoDataLocality() { return accessIDtoDataLocality_; }

private:
  /// @brief use the data locality metric to rank the fields in a stencil based on how much they
  /// would benefit from caching
  void computeOptimalFields() {
    // Call the improved metric calculator here TODO
    // dataLocality holds <numreads, numwrites>
    auto dataLocality =
        computeReadWriteAccessesMetricIndividually(instantiation_, *multiStagePrt_, config_);

    for(const auto& stagePtr : multiStagePrt_->getStages()) {
      for(const Field& field : stagePtr->getFields()) {
        auto findField = accessIDtoDataLocality_.find(field.AccessID);

        // We already checked this field
        if(findField != accessIDtoDataLocality_.end())
          continue;

        // Only ij-cache fields that are not horizontally pointwise and are vertically pointwise
        if(!(field.Extent.isVerticalPointwise()) || (field.Extent.isHorizontalPointwise()))
          continue;

        // This is caching non-temporary fields
        if(instantiation_->isTemporaryField(field.AccessID))
          continue;

        sortedAccesses_.emplace_back(field.AccessID, 1);

        int cachedreadwrites = (*dataLocality.find(field.AccessID)).second.total();
//        std::cout << std::endl
//                  << "In the Field (before subtracting): "
//                  << instantiation_->getNameFromAccessID(field.AccessID) << " we have "
//                  << dataLocality.find(field.AccessID)->second.numReads
//                  << " read operations\n and we have "
//                  << dataLocality.find(field.AccessID)->second.numWrites << " wirte operations"
//                  << std::endl;

        // We need one more read and one more write due to cache-filling
        if(checkReadBeforeWrite(field.AccessID))
          cachedreadwrites -= 2;
        // We need one more write to flush the cache back to the variable
        cachedreadwrites--;

        accessIDtoDataLocality_.emplace(field.AccessID, cachedreadwrites);
      }
    }

    for(auto& performanceMesure : accessIDtoDataLocality_) {
      sortedAccesses_.emplace_back(performanceMesure.first, performanceMesure.second);
      std::sort(sortedAccesses_.begin(), sortedAccesses_.end());
    }
  }

  /// @brief iterate through the multistage and find the first read-statement for evey cached
  /// variable and insert the cache fill before that statement
  void addFillerStages() {
    for(int i = 0; i < std::min((int)sortedAccesses_.size(), config_.SMemMaxFields); ++i) {
      if(sortedAccesses_[i].dataLocalityGain > 0) {
        int oldID = sortedAccesses_[i].accessID;

        // Create new temporary field and register in the instantiation
        int newID = instantiation_->nextUID();

        instantiation_->setAccessIDNamePairOfField(newID, dawn::format("tempCache%i", i), true);

        // Rename all the fields in this multistage
        multiStagePrt_->renameAllOccurrences(oldID, newID);

        oldIDtoNewID.emplace(oldID, newID);
        Cache& cache = multiStagePrt_->setCache(Cache::IJ, Cache::local, newID);
        idToCache_.emplace(oldID, cache);
      }

      // Create the cache-filler stage
      std::vector<int> oldIDs;
      std::vector<int> newIDs;
      for(auto& idPair : oldIDtoNewID) {
        bool hasReadBeforeWrite = checkReadBeforeWrite(idPair.first);
        if(hasReadBeforeWrite) {
          oldIDs.push_back(idPair.first);
          newIDs.push_back(idPair.second);
        }
      }
      if(oldIDs.size() > 0) {
        auto stageBegin = multiStagePrt_->getStages().begin();

        DoMethod& method = (stageBegin)->get()->getSingleDoMethod();
        std::shared_ptr<Stage> cacheFillStage =
            createAssignmentStage(method.getInterval(), newIDs, oldIDs);
        multiStagePrt_->getStages().insert(stageBegin, std::move(cacheFillStage));
      }
    }
  }

  void addFlushStages() {
    std::vector<int> oldIDs;
    std::vector<int> newIDs;
    for(auto& idPair : oldIDtoNewID) {
      oldIDs.push_back(idPair.first);
      newIDs.push_back(idPair.second);
    }

    if(oldIDs.size() > 0) {
      auto stageEnd = multiStagePrt_->getStages().end();

      DoMethod& method = (--stageEnd)->get()->getSingleDoMethod();

      std::shared_ptr<Stage> cacheFlushStage =
          createAssignmentStage(method.getInterval(), oldIDs, newIDs);

      // Insert the new stage at the found location
      multiStagePrt_->getStages().insert(++stageEnd, std::move(cacheFlushStage));
    }
  }

  std::shared_ptr<Stage> createAssignmentStage(Interval& interval,
                                               const std::vector<int>& assignmentIDs,
                                               const std::vector<int>& assigneeIDs) {
    // Add the cache Flush stage
    std::shared_ptr<Stage> assignmentStage = std::make_shared<Stage>(
        instantiation_, multiStagePrt_, instantiation_->nextUID(), interval);
    std::unique_ptr<DoMethod> domethod = make_unique<DoMethod>(assignmentStage.get(), interval);
    domethod->getStatementAccessesPairs().clear();

    for(int i = 0; i < assignmentIDs.size(); ++i) {
      int assignmentID = assignmentIDs[i];
      int assigneeID = assigneeIDs[i];
      addAssignmentToDoMethod(domethod, assignmentID, assigneeID);
    }

    // Add the single do method to the new Stage
    assignmentStage->getDoMethods().clear();
    assignmentStage->addDoMethod(domethod);
    assignmentStage->update();

    return assignmentStage;
  }

  void addAssignmentToDoMethod(std::unique_ptr<DoMethod>& domethod, int assignmentID,
                               int assigneeID) {
    // Create the StatementAccessPair of the assignment with the new and old variables
    std::shared_ptr<FieldAccessExpr> fa_assignee =
        std::make_shared<FieldAccessExpr>(instantiation_->getNameFromAccessID(assigneeID));
    std::shared_ptr<FieldAccessExpr> fa_assignment =
        std::make_shared<FieldAccessExpr>(instantiation_->getNameFromAccessID(assignmentID));
    std::shared_ptr<AssignmentExpr> assignmentExpression =
        std::make_shared<AssignmentExpr>(fa_assignment, fa_assignee, "=");
    std::shared_ptr<ExprStmt> expAssignment = std::make_shared<ExprStmt>(assignmentExpression);
    std::shared_ptr<Statement> assignmentStatement =
        std::make_shared<Statement>(expAssignment, nullptr);
    std::shared_ptr<StatementAccessesPair> pair =
        std::make_shared<StatementAccessesPair>(assignmentStatement);
    std::shared_ptr<Accesses> newAccess = std::make_shared<Accesses>();
    newAccess->addWriteExtent(assignmentID, Extents({0, 0, 0}));
    newAccess->addReadExtent(assigneeID, Extents({0, 0, 0}));
    pair->setAccesses(newAccess);
    domethod->getStatementAccessesPairs().push_back(pair);

    // Add the new expressions to the map
    instantiation_->getExprToAccessIDMap().emplace(fa_assignment, assignmentID);
    instantiation_->getExprToAccessIDMap().emplace(fa_assignee, assigneeID);
  }

  bool checkReadBeforeWrite(int id) {
    for(auto stageItGlob = multiStagePrt_->getStages().begin();
        stageItGlob != multiStagePrt_->getStages().end(); ++stageItGlob) {
      DoMethod& doMethod = (**stageItGlob).getSingleDoMethod();
      for(int stmtIndex = 0; stmtIndex < doMethod.getStatementAccessesPairs().size(); ++stmtIndex) {
        const std::shared_ptr<StatementAccessesPair>& stmtAccessesPair =
            doMethod.getStatementAccessesPairs()[stmtIndex];
        // Find first if this statement has a read
        auto readAccessIterator = stmtAccessesPair->getCallerAccesses()->getReadAccesses().find(id);
        if(readAccessIterator != stmtAccessesPair->getCallerAccesses()->getReadAccesses().end()) {
          return true;
        }
        // If we did not find a read statement so far, we have  a write first and do not need to
        // fill the cache
        auto wirteAccessIterator =
            stmtAccessesPair->getCallerAccesses()->getWriteAccesses().find(id);
        if(wirteAccessIterator != stmtAccessesPair->getCallerAccesses()->getWriteAccesses().end()) {
          return false;
        }
      }
    }
    return false;
  }

  MultiStage* multiStagePrt_;
  std::unordered_map<int, int> accessIDtoDataLocality_;
  std::unordered_map<int, int> oldIDtoNewID;
  std::vector<accessIDaccessMetric> sortedAccesses_;
  std::vector<int> fieldsToCheck_;
  HardwareConfig config_;
  StencilInstantiation* instantiation_;

  std::unordered_map<int, Cache> idToCache_;
};

PassSetNonTempCaches::PassSetNonTempCaches() : Pass("PassSetNonTempCaches") {}

bool dawn::PassSetNonTempCaches::run(dawn::StencilInstantiation* stencilInstantiation) {

  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    const Stencil& stencil = *stencilPtr;

    if(context->getOptions().ReportPassSetNonTempCaches) {
      std::cout << "\nPASS: " << getName() << ": In stencil: " << stencilInstantiation->getName()
                << " : ";
    }
    if(context->getOptions().UseNonTempCaches) {
      // Cache normal fields within multistages
      for(auto& multiStagePtr : stencil.getMultiStages()) {
        GlobalFieldCacher organizer(multiStagePtr.get(), stencilInstantiation);
        organizer.process();
        if(context->getOptions().ReportPassSetNonTempCaches) {
          for(const auto& idToCache : organizer.getIDtocacheMap())
            std::cout << stencilInstantiation->getOriginalNameFromAccessID(idToCache.first) << ":"
                      << idToCache.second.getCacheTypeAsString() << ":"
                      << idToCache.second.getCacheIOPolicyAsString()
                      << " ,expected perfomance gain: "
                      << organizer.getaccessIDtoDataLocality().find(idToCache.first)->second
                      << std::endl;
        }
      }
    }
  }

  return true;
}

} // namespace dawn
