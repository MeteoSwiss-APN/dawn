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
#include "dawn/IIR/Cache.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassDataLocalityMetric.h"
#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/SIR/ASTExpr.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>
#include <set>
#include <vector>

namespace dawn {

struct AcessIDTolocalityMetric {
  int accessID;
  int dataLocalityGain;
  bool operator<(const AcessIDTolocalityMetric& rhs) {
    return dataLocalityGain < rhs.dataLocalityGain;
  }
};

struct NameToImprovementMetric {
  std::string name;
  iir::Cache cache;
  int dataLocalityImprovement;
};

/// @brief The GlobalFieldCacher class handles the caching for a given Multistage
class GlobalFieldCacher {
public:
  /// @param[in, out]  msprt   Pointer to the multistage to handle
  /// @param[in, out]  si      Stencil Instanciation [ISIR] holding all the Stencils
  GlobalFieldCacher(const std::unique_ptr<iir::MultiStage>& msptr,
                    const std::shared_ptr<iir::StencilInstantiation>& si)
      : multiStagePrt_(msptr), instantiation_(si) {}

  /// @brief Entry method for the pass: processes a given multistage and applies all changes
  /// required
  void process() {
    computeOptimalFields();
    addFillerStages();
    addFlushStages();
  }

  const std::vector<NameToImprovementMetric>& getOriginalNameToCache() const {
    return originalNameToCache_;
  }

private:
  /// @brief Use the data locality metric to rank the fields in a stencil based on how much they
  /// would benefit from caching
  void computeOptimalFields() {
    auto dataLocality =
        computeReadWriteAccessesMetricPerAccessID(instantiation_->getIIR().get(), *multiStagePrt_);

    for(const auto& stagePtr : multiStagePrt_->getChildren()) {
      for(const auto& fieldPair : stagePtr->getFields()) {
        const iir::Field& field = fieldPair.second;
        auto iter = accessIDToDataLocality_.find(field.getAccessID());

        // We already checked this field
        if(iter != accessIDToDataLocality_.end())
          continue;

        // Only ij-cache fields that are not horizontally pointwise and are vertically pointwise
        if(!field.getExtents().isVerticalPointwise() || field.getExtents().isHorizontalPointwise())
          continue;

        // This is caching non-temporary fields
        if(instantiation_->getIIR()->getMetaData()->isTemporaryField(field.getAccessID()))
          continue;

        int cachedReadAndWrites = dataLocality.find(field.getAccessID())->second.totalAccesses();

        // We check how much the cache-filling costs: 1 Read and if we fill from Memory
        if(checkReadBeforeWrite(field.getAccessID()))
          cachedReadAndWrites--;
        // We need one more write to flush the cache back to the variable
        if(!checkReadOnlyAccess(field.getAccessID()))
          cachedReadAndWrites--;

        if(cachedReadAndWrites > 0)
          accessIDToDataLocality_.emplace(field.getAccessID(), cachedReadAndWrites);
      }
    }

    for(auto& AcessIDMetricPair : accessIDToDataLocality_) {
      sortedAccesses_.emplace_back(
          AcessIDTolocalityMetric{AcessIDMetricPair.first, AcessIDMetricPair.second});
      std::sort(sortedAccesses_.begin(), sortedAccesses_.end());
    }
  }

  /// @brief For each chached variable, check if we need a filler-stage (if we find a
  /// read-before-write) and add it if necessary, renames all occurances and sets the cache for the
  /// temporary variable newly created
  ///
  /// We always fill to the extent of the given multistage and we only add one stage to fill all the
  /// variables to reduce synchronisation overhead
  void addFillerStages() {
    int numVarsToBeCached =
        std::min((int)sortedAccesses_.size(),
                 instantiation_->getIIR()->getHardwareConfiguration().SMemMaxFields);
    for(int i = 0; i < numVarsToBeCached; ++i) {
      int oldID = sortedAccesses_[i].accessID;

      // Create new temporary field and register in the instantiation
      int newID = instantiation_->getIIR()->getMetaData()->nextUID();

      instantiation_->getIIR()->getMetaData()->setAccessIDNamePairOfField(
          newID, "__tmp_cache_" + std::to_string(i), true);

      // Rename all the fields in this multistage
      multiStagePrt_->renameAllOccurrences(oldID, newID);

      oldAccessIDtoNewAccessID_.emplace(oldID, newID);
      iir::Cache& cache = multiStagePrt_->setCache(iir::Cache::IJ, iir::Cache::local, newID);
      originalNameToCache_.emplace_back(
          NameToImprovementMetric{instantiation_->getIIR()->getMetaData()->getOriginalNameFromAccessID(oldID), cache,
                                  accessIDToDataLocality_.find(oldID)->second});
      instantiation_->getIIR()->getMetaData()->insertCachedVariable(oldID);
      instantiation_->getIIR()->getMetaData()->insertCachedVariable(newID);
    }

    // Create the cache-filler stage
    std::vector<int> oldAccessIDs;
    std::vector<int> newAccessIDs;
    for(auto& idPair : oldAccessIDtoNewAccessID_) {
      bool hasReadBeforeWrite = checkReadBeforeWrite(idPair.second);
      if(hasReadBeforeWrite) {
        oldAccessIDs.push_back(idPair.first);
        newAccessIDs.push_back(idPair.second);
      }
    }
    if(!oldAccessIDs.empty()) {
      auto stageBegin = multiStagePrt_->childrenBegin();

      std::unique_ptr<iir::Stage> cacheFillStage =
          createAssignmentStage(multiStagePrt_->getEnclosingInterval(), newAccessIDs, oldAccessIDs);
      multiStagePrt_->insertChild(stageBegin, std::move(cacheFillStage));
    }
  }

  /// @brief We create one stage at the end of the multistage that flushes back all the variables
  /// that we cached in temporaries
  void addFlushStages() {
    std::vector<int> oldAccessIDs;
    std::vector<int> newAccessIDs;
    for(auto& idPair : oldAccessIDtoNewAccessID_) {
      if(!checkReadOnlyAccess(idPair.second)) {
        oldAccessIDs.push_back(idPair.first);
        newAccessIDs.push_back(idPair.second);
      }
    }

    if(!oldAccessIDs.empty()) {
      auto stageEnd = multiStagePrt_->childrenEnd();

      std::unique_ptr<iir::Stage> cacheFlushStage =
          createAssignmentStage(multiStagePrt_->getEnclosingInterval(), oldAccessIDs, newAccessIDs);

      // Insert the new stage at the found location
      multiStagePrt_->insertChild(stageEnd, std::move(cacheFlushStage));
    }
  }

  /// @brief Creates the stage in which assignment happens (fill and flush)
  std::unique_ptr<iir::Stage> createAssignmentStage(const iir::Interval& interval,
                                                    const std::vector<int>& assignmentIDs,
                                                    const std::vector<int>& assigneeIDs) {
    // Add the cache Flush stage
    std::unique_ptr<iir::Stage> assignmentStage =
        make_unique<iir::Stage>(instantiation_->getIIR().get(),
                                instantiation_->getIIR()->getMetaData()->nextUID(), interval);
    iir::Stage::DoMethodSmartPtr_t domethod = make_unique<iir::DoMethod>(interval);
    domethod->clearChildren();

    for(int i = 0; i < assignmentIDs.size(); ++i) {
      int assignmentID = assignmentIDs[i];
      int assigneeID = assigneeIDs[i];
      addAssignmentToDoMethod(domethod, assignmentID, assigneeID);
    }

    // Add the single do method to the new Stage
    assignmentStage->clearChildren();
    assignmentStage->addDoMethod(domethod);
    assignmentStage->update(iir::NodeUpdateType::level);

    return assignmentStage;
  }

  ///@brief Add the assignment operator of two unique id's to a given domethod
  void addAssignmentToDoMethod(const iir::Stage::DoMethodSmartPtr_t& domethod, int assignmentID,
                               int assigneeID) {
    // Create the StatementAccessPair of the assignment with the new and old variables
    auto fa_assignee = std::make_shared<FieldAccessExpr>(
        instantiation_->getIIR()->getMetaData()->getNameFromAccessID(assigneeID));
    auto fa_assignment = std::make_shared<FieldAccessExpr>(
        instantiation_->getIIR()->getMetaData()->getNameFromAccessID(assignmentID));
    auto assignmentExpression = std::make_shared<AssignmentExpr>(fa_assignment, fa_assignee, "=");
    auto expAssignment = std::make_shared<ExprStmt>(assignmentExpression);
    auto assignmentStatement = std::make_shared<Statement>(expAssignment, nullptr);
    auto pair = make_unique<iir::StatementAccessesPair>(assignmentStatement);
    auto newAccess = std::make_shared<iir::Accesses>();
    newAccess->addWriteExtent(assignmentID, iir::Extents(Array3i{{0, 0, 0}}));
    newAccess->addReadExtent(assigneeID, iir::Extents(Array3i{{0, 0, 0}}));
    pair->setAccesses(newAccess);
    domethod->insertChild(std::move(pair));

    // Add the new expressions to the map
    instantiation_->getIIR()->getMetaData()->mapExprToAccessID(fa_assignment, assignmentID);
    instantiation_->getIIR()->getMetaData()->mapExprToAccessID(fa_assignee, assigneeID);
  }

  /// @brief Checks if there is a read operation before the first write operation in the given
  /// multistage.
  /// @param[in]    AccessID  original FieldAccessID of the field to check
  /// @return true if there is a read-access before the first write access
  bool checkReadBeforeWrite(int AccessID) {

    for(const auto& stmtAccessesPair :
        iterateIIROver<iir::StatementAccessesPair>(*multiStagePrt_)) {

      // Find first if this statement has a read
      auto readAccessIterator =
          stmtAccessesPair->getCallerAccesses()->getReadAccesses().find(AccessID);
      if(readAccessIterator != stmtAccessesPair->getCallerAccesses()->getReadAccesses().end()) {
        return true;
      }
      // If we did not find a read statement so far, we have  a write first and do not need to
      // fill the cache
      auto wirteAccessIterator =
          stmtAccessesPair->getCallerAccesses()->getWriteAccesses().find(AccessID);
      if(wirteAccessIterator != stmtAccessesPair->getCallerAccesses()->getWriteAccesses().end()) {
        return false;
      }
    }
    return false;
  }

  bool checkReadOnlyAccess(int AccessID) {

    for(const auto& stmtAccessesPair :
        iterateIIROver<iir::StatementAccessesPair>(*multiStagePrt_)) {

      // If we find a write-statement we exit
      auto wirteAccessIterator =
          stmtAccessesPair->getCallerAccesses()->getWriteAccesses().find(AccessID);
      if(wirteAccessIterator != stmtAccessesPair->getCallerAccesses()->getWriteAccesses().end()) {
        return false;
      }
    }
    return true;
  }

  const std::unique_ptr<iir::MultiStage>& multiStagePrt_;
  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;

  std::unordered_map<int, int> accessIDToDataLocality_;
  std::unordered_map<int, int> oldAccessIDtoNewAccessID_;
  std::vector<AcessIDTolocalityMetric> sortedAccesses_;

  std::vector<NameToImprovementMetric> originalNameToCache_;
};

PassSetNonTempCaches::PassSetNonTempCaches() : Pass("PassSetNonTempCaches") {}

bool dawn::PassSetNonTempCaches::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  for(const auto& stencilPtr : stencilInstantiation->getIIR()->getChildren()) {
    const iir::Stencil& stencil = *stencilPtr;

    std::vector<NameToImprovementMetric> allCachedFields;
    if(stencilInstantiation->getIIR()->getOptions().UseNonTempCaches) {
      for(const auto& multiStagePtr : stencil.getChildren()) {
        GlobalFieldCacher organizer(multiStagePtr, stencilInstantiation);
        organizer.process();
        if(stencilInstantiation->getIIR()->getOptions().ReportPassSetNonTempCaches) {
          for(const auto& nametoCache : organizer.getOriginalNameToCache())
            allCachedFields.push_back(nametoCache);
        }
      }
    }
    // Output
    if(stencilInstantiation->getIIR()->getOptions().ReportPassSetNonTempCaches) {
      std::sort(allCachedFields.begin(), allCachedFields.end(),
                [](const NameToImprovementMetric& lhs, const NameToImprovementMetric& rhs) {
                  return lhs.name < rhs.name;
                });
      std::cout << "\nPASS: " << getName() << ": "
                << stencilInstantiation->getIIR()->getMetaData()->getName() << " :";
      for(const auto& nametoCache : allCachedFields) {
        std::cout << " Cached: " << nametoCache.name
                  << " : Type: " << nametoCache.cache.getCacheTypeAsString() << ":"
                  << nametoCache.cache.getCacheIOPolicyAsString();
      }
      if(allCachedFields.size() == 0) {
        std::cout << " no fields cached";
      }
      std::cout << std::endl;
    }
  }

  return true;
}

} // namespace dawn
