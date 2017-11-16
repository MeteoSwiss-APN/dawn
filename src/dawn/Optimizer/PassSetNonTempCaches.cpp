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

struct nameToImprovementMetric {
  nameToImprovementMetric(std::string name_, Cache cache_, int metric)
      : name(name_), cache(cache_), dataLocalityImprovement(metric) {}
  std::string name;
  Cache cache;
  int dataLocalityImprovement;
};

/// @brief The GlobalFieldCacher class handles the caching for a given Multistage
/// @param[in, out]  msprt   Pointer to the multistage to handle
/// @param[in, out]  si      Stencil Instanciation [ISIR] holding all the Stencils
class GlobalFieldCacher {
public:
  GlobalFieldCacher(MultiStage* msptr, StencilInstantiation* si)
      : multiStagePrt_(msptr), instantiation_(si) {}

  /// @brief entry method for the pass: processes a given multistage and applies all changes
  /// required
  void process() {
    computeOptimalFields();
    addFillerStages();
    addFlushStages();
  }

  std::vector<nameToImprovementMetric> getoriginalNameToCache() { return originalNameToCache_; }

private:
  /// @brief use the data locality metric to rank the fields in a stencil based on how much they
  /// would benefit from caching
  void computeOptimalFields() {
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

        int cachedreadwrites = dataLocality.find(field.AccessID)->second.total();

        // We check how much the cache-filling costs: 1 Read and if we fill from Memory
        if(checkReadBeforeWrite(field.AccessID))
          cachedreadwrites--;
        // We need one more write to flush the cache back to the variable
        if(!checkReadOnlyAccess(field.AccessID))
          cachedreadwrites--;

        if(cachedreadwrites > 0)
          accessIDtoDataLocality_.emplace(field.AccessID, cachedreadwrites);
      }
    }

    for(auto& performanceMesure : accessIDtoDataLocality_) {
      sortedAccesses_.emplace_back(performanceMesure.first, performanceMesure.second);
      std::sort(sortedAccesses_.begin(), sortedAccesses_.end());
    }
  }

  /// @brief for each chached variable, check if we need a filler-stage (if we find a
  /// read-before-write) and add it if necessary, renames all occurances and sets the cache for the
  /// temporary variable newly created
  ///
  /// We always fill to the extent of the given multistage and we only add one stage to fill all the
  /// variables to reduce synchronisation overhead
  void addFillerStages() {
    for(int i = 0; i < std::min((int)sortedAccesses_.size(), config_.SMemMaxFields); ++i) {
      int oldID = sortedAccesses_[i].accessID;

      // Create new temporary field and register in the instantiation
      int newID = instantiation_->nextUID();

      instantiation_->setAccessIDNamePairOfField(newID, dawn::format("__tmp_cache_%i", i), true);

      // Rename all the fields in this multistage
      multiStagePrt_->renameAllOccurrences(oldID, newID);

      oldIDtoNewID.emplace(oldID, newID);
      Cache& cache = multiStagePrt_->setCache(Cache::IJ, Cache::local, newID);
      originalNameToCache_.emplace_back(instantiation_->getOriginalNameFromAccessID(oldID), cache,
                                        accessIDtoDataLocality_.find(oldID)->second);
    }

    // Create the cache-filler stage
    std::vector<int> oldIDs;
    std::vector<int> newIDs;
    for(auto& idPair : oldIDtoNewID) {
      bool hasReadBeforeWrite = checkReadBeforeWrite(idPair.second);
      if(hasReadBeforeWrite) {
        oldIDs.push_back(idPair.first);
        newIDs.push_back(idPair.second);
      }
    }
    if(oldIDs.size() > 0) {
      auto stageBegin = multiStagePrt_->getStages().begin();

      std::shared_ptr<Stage> cacheFillStage =
          createAssignmentStage(multiStagePrt_->getEnclosingInterval(), newIDs, oldIDs);
      multiStagePrt_->getStages().insert(stageBegin, std::move(cacheFillStage));
    }
  }

  /// @brief we create one stage at the end of the multistage that flushes back all the variables
  /// that we cached in temporaries
  void addFlushStages() {
    std::vector<int> oldIDs;
    std::vector<int> newIDs;
    for(auto& idPair : oldIDtoNewID) {
      if(!checkReadOnlyAccess(idPair.second)) {
        oldIDs.push_back(idPair.first);
        newIDs.push_back(idPair.second);
      }
    }

    if(oldIDs.size() > 0) {
      auto stageEnd = multiStagePrt_->getStages().end();

      std::shared_ptr<Stage> cacheFlushStage =
          createAssignmentStage(multiStagePrt_->getEnclosingInterval(), oldIDs, newIDs);

      // Insert the new stage at the found location
      multiStagePrt_->getStages().insert(stageEnd, std::move(cacheFlushStage));
    }
  }

  /// @brief creates the stage in which assignment happens (fill and flush)
  std::shared_ptr<Stage> createAssignmentStage(const Interval& interval,
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

  ///@brief add the assignment operator of two unique id's to a given domethod
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

  /// @brief checks if there is a read operation before the first write operation in the given
  /// multistage.
  /// @param[in]    id  original FieldAccessID of the field to check
  /// @return           true if there is a read-access before the first write access
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

  bool checkReadOnlyAccess(int id) {
      for(auto stageItGlob = multiStagePrt_->getStages().begin();
          stageItGlob != multiStagePrt_->getStages().end(); ++stageItGlob) {
        DoMethod& doMethod = (**stageItGlob).getSingleDoMethod();
        for(int stmtIndex = 0; stmtIndex < doMethod.getStatementAccessesPairs().size(); ++stmtIndex) {
          const std::shared_ptr<StatementAccessesPair>& stmtAccessesPair =
              doMethod.getStatementAccessesPairs()[stmtIndex];

          // If we find a write-statement we exit
          auto wirteAccessIterator =
              stmtAccessesPair->getCallerAccesses()->getWriteAccesses().find(id);
          if(wirteAccessIterator != stmtAccessesPair->getCallerAccesses()->getWriteAccesses().end()) {
            return false;
          }
        }
      }
      return true;
  }

  MultiStage* multiStagePrt_;
  HardwareConfig config_;
  StencilInstantiation* instantiation_;

  std::unordered_map<int, int> accessIDtoDataLocality_;
  std::unordered_map<int, int> oldIDtoNewID;
  std::vector<accessIDaccessMetric> sortedAccesses_;

  std::vector<nameToImprovementMetric> originalNameToCache_;
};

PassSetNonTempCaches::PassSetNonTempCaches() : Pass("PassSetNonTempCaches") {}

bool dawn::PassSetNonTempCaches::run(dawn::StencilInstantiation* stencilInstantiation) {

  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    const Stencil& stencil = *stencilPtr;

    std::vector<nameToImprovementMetric> allCachedFields;
    if(context->getOptions().UseNonTempCaches) {
      for(auto& multiStagePtr : stencil.getMultiStages()) {
        GlobalFieldCacher organizer(multiStagePtr.get(), stencilInstantiation);
        organizer.process();
        if(context->getOptions().ReportPassSetNonTempCaches) {
          for(const auto& nametoCache : organizer.getoriginalNameToCache())
            allCachedFields.push_back(nametoCache);
        }
      }
    }
    // Output
    if(context->getOptions().ReportPassSetNonTempCaches) {
      std::sort(allCachedFields.begin(), allCachedFields.end(),
                [](const nameToImprovementMetric& lhs, const nameToImprovementMetric& rhs) {
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
