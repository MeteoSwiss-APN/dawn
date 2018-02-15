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

struct AcessIDTolocalityMetric {
  int accessID;
  int dataLocalityGain;
  bool operator<(const AcessIDTolocalityMetric& rhs) {
    return dataLocalityGain < rhs.dataLocalityGain;
  }
};

struct NameToImprovementMetric {
  std::string name;
  Cache cache;
  int dataLocalityImprovement;
};

/// @brief The GlobalFieldCacher class handles the caching for a given Multistage
class GlobalFieldCacher {
public:
  /// @param[in, out]  msprt   Pointer to the multistage to handle
  /// @param[in, out]  si      Stencil Instanciation [ISIR] holding all the Stencils
  GlobalFieldCacher(MultiStage* msptr, std::shared_ptr<StencilInstantiation> si)
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
    auto dataLocality = computeReadWriteAccessesMetricPerAccessID(instantiation_, *multiStagePrt_);

    for(const auto& stagePtr : multiStagePrt_->getStages()) {
      for(const Field& field : stagePtr->getFields()) {
        auto iter = accessIDToDataLocality_.find(field.getAccessID());

        // We already checked this field
        if(iter != accessIDToDataLocality_.end())
          continue;

        // Only ij-cache fields that are not horizontally pointwise and are vertically pointwise
        if(!field.getExtents().isVerticalPointwise() || field.getExtents().isHorizontalPointwise())
          continue;

        // This is caching non-temporary fields
        if(instantiation_->isTemporaryField(field.getAccessID()))
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
                 instantiation_->getOptimizerContext()->getHardwareConfiguration().SMemMaxFields);
    for(int i = 0; i < numVarsToBeCached; ++i) {
      int oldID = sortedAccesses_[i].accessID;

      // Create new temporary field and register in the instantiation
      int newID = instantiation_->nextUID();

      instantiation_->setAccessIDNamePairOfField(newID, "__tmp_cache_" + std::to_string(i), true);

      // Rename all the fields in this multistage
      multiStagePrt_->renameAllOccurrences(oldID, newID);

      oldAccessIDtoNewAccessID_.emplace(oldID, newID);
      Cache& cache = multiStagePrt_->setCache(Cache::IJ, Cache::local, newID);
      originalNameToCache_.emplace_back(
          NameToImprovementMetric{instantiation_->getOriginalNameFromAccessID(oldID), cache,
                                  accessIDToDataLocality_.find(oldID)->second});
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
      auto stageBegin = multiStagePrt_->getStages().begin();

      std::shared_ptr<Stage> cacheFillStage =
          createAssignmentStage(multiStagePrt_->getEnclosingInterval(), newAccessIDs, oldAccessIDs);
      multiStagePrt_->getStages().insert(stageBegin, std::move(cacheFillStage));
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
      auto stageEnd = multiStagePrt_->getStages().end();

      std::shared_ptr<Stage> cacheFlushStage =
          createAssignmentStage(multiStagePrt_->getEnclosingInterval(), oldAccessIDs, newAccessIDs);

      // Insert the new stage at the found location
      multiStagePrt_->getStages().insert(stageEnd, std::move(cacheFlushStage));
    }
  }

  /// @brief Creates the stage in which assignment happens (fill and flush)
  std::shared_ptr<Stage> createAssignmentStage(const Interval& interval,
                                               const std::vector<int>& assignmentIDs,
                                               const std::vector<int>& assigneeIDs) {
    // Add the cache Flush stage
    std::shared_ptr<Stage> assignmentStage = std::make_shared<Stage>(
        *instantiation_, multiStagePrt_, instantiation_->nextUID(), interval);
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

  ///@brief Add the assignment operator of two unique id's to a given domethod
  void addAssignmentToDoMethod(std::unique_ptr<DoMethod>& domethod, int assignmentID,
                               int assigneeID) {
    // Create the StatementAccessPair of the assignment with the new and old variables
    auto fa_assignee =
        std::make_shared<FieldAccessExpr>(instantiation_->getNameFromAccessID(assigneeID));
    auto fa_assignment =
        std::make_shared<FieldAccessExpr>(instantiation_->getNameFromAccessID(assignmentID));
    auto assignmentExpression = std::make_shared<AssignmentExpr>(fa_assignment, fa_assignee, "=");
    auto expAssignment = std::make_shared<ExprStmt>(assignmentExpression);
    auto assignmentStatement = std::make_shared<Statement>(expAssignment, nullptr);
    auto pair = std::make_shared<StatementAccessesPair>(assignmentStatement);
    auto newAccess = std::make_shared<Accesses>();
    newAccess->addWriteExtent(assignmentID, Extents(Array3i{{0, 0, 0}}));
    newAccess->addReadExtent(assigneeID, Extents(Array3i{{0, 0, 0}}));
    pair->setAccesses(newAccess);
    domethod->getStatementAccessesPairs().push_back(pair);

    // Add the new expressions to the map
    instantiation_->mapExprToAccessID(fa_assignment, assignmentID);
    instantiation_->mapExprToAccessID(fa_assignee, assigneeID);
  }

  /// @brief Checks if there is a read operation before the first write operation in the given
  /// multistage.
  /// @param[in]    AccessID  original FieldAccessID of the field to check
  /// @return true if there is a read-access before the first write access
  bool checkReadBeforeWrite(int AccessID) {
    for(auto stageItGlob = multiStagePrt_->getStages().begin();
        stageItGlob != multiStagePrt_->getStages().end(); ++stageItGlob) {
      DoMethod& doMethod = (**stageItGlob).getSingleDoMethod();
      for(int stmtIndex = 0; stmtIndex < doMethod.getStatementAccessesPairs().size(); ++stmtIndex) {
        const std::shared_ptr<StatementAccessesPair>& stmtAccessesPair =
            doMethod.getStatementAccessesPairs()[stmtIndex];

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
    }
    return false;
  }

  bool checkReadOnlyAccess(int AccessID) {
    for(auto stageItGlob = multiStagePrt_->getStages().begin();
        stageItGlob != multiStagePrt_->getStages().end(); ++stageItGlob) {
      DoMethod& doMethod = (**stageItGlob).getSingleDoMethod();
      for(int stmtIndex = 0; stmtIndex < doMethod.getStatementAccessesPairs().size(); ++stmtIndex) {
        const std::shared_ptr<StatementAccessesPair>& stmtAccessesPair =
            doMethod.getStatementAccessesPairs()[stmtIndex];
        // If we find a write-statement we exit
        auto wirteAccessIterator =
            stmtAccessesPair->getCallerAccesses()->getWriteAccesses().find(AccessID);
        if(wirteAccessIterator != stmtAccessesPair->getCallerAccesses()->getWriteAccesses().end()) {
          return false;
        }
      }
    }
    return true;
  }

  MultiStage* multiStagePrt_;
  std::shared_ptr<StencilInstantiation> instantiation_;

  std::unordered_map<int, int> accessIDToDataLocality_;
  std::unordered_map<int, int> oldAccessIDtoNewAccessID_;
  std::vector<AcessIDTolocalityMetric> sortedAccesses_;

  std::vector<NameToImprovementMetric> originalNameToCache_;
};

PassSetNonTempCaches::PassSetNonTempCaches() : Pass("PassSetNonTempCaches") {}

bool dawn::PassSetNonTempCaches::run(
    const std::shared_ptr<StencilInstantiation>& stencilInstantiation) {

  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    const Stencil& stencil = *stencilPtr;

    std::vector<NameToImprovementMetric> allCachedFields;
    if(context->getOptions().UseNonTempCaches) {
      for(auto& multiStagePtr : stencil.getMultiStages()) {
        GlobalFieldCacher organizer(multiStagePtr.get(), stencilInstantiation);
        organizer.process();
        if(context->getOptions().ReportPassSetNonTempCaches) {
          for(const auto& nametoCache : organizer.getOriginalNameToCache())
            allCachedFields.push_back(nametoCache);
        }
      }
    }
    // Output
    if(context->getOptions().ReportPassSetNonTempCaches) {
      std::sort(allCachedFields.begin(), allCachedFields.end(),
                [](const NameToImprovementMetric& lhs, const NameToImprovementMetric& rhs) {
                  return lhs.name < rhs.name;
                });
      std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName() << " :";
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
