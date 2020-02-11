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
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/Cache.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassDataLocalityMetric.h"
#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>
#include <set>
#include <vector>

namespace dawn {

struct AccessIDToLocalityMetric {
  int accessID;
  int dataLocalityGain;
  bool operator<(const AccessIDToLocalityMetric& rhs) const {
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
  const std::unique_ptr<iir::MultiStage>& multiStagePrt_;
  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;
  iir::StencilMetaInformation& metadata_;
  OptimizerContext& context_;
  std::unordered_map<int, int> accessIDToDataLocality_;
  std::unordered_map<int, int> oldAccessIDtoNewAccessID_;
  std::vector<AccessIDToLocalityMetric> sortedAccesses_;
  std::vector<NameToImprovementMetric> originalNameToCache_;

public:
  /// @param[in, out]  msprt   Pointer to the multistage to handle
  /// @param[in, out]  si      Stencil Instantiation [ISIR] holding all the Stencils
  GlobalFieldCacher(const std::unique_ptr<iir::MultiStage>& msptr,
                    const std::shared_ptr<iir::StencilInstantiation>& si, OptimizerContext& context)
      : multiStagePrt_(msptr), instantiation_(si), metadata_(si->getMetaData()), context_(context) {
  }

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
        computeReadWriteAccessesMetricPerAccessID(instantiation_, context_, *multiStagePrt_);

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
        if(metadata_.isAccessType(iir::FieldAccessType::StencilTemporary, field.getAccessID()))
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
          AccessIDToLocalityMetric{AcessIDMetricPair.first, AcessIDMetricPair.second});
      std::sort(sortedAccesses_.begin(), sortedAccesses_.end());
    }
  }

  /// @brief For each cached variable, check if we need a filler-stage (if we find a
  /// read-before-write) and add it if necessary, renames all occurrences and sets the cache for the
  /// temporary variable newly created
  ///
  /// We always fill to the extent of the given multistage and we only add one stage to fill all the
  /// variables to reduce synchronisation overhead
  void addFillerStages() {
    int numVarsToBeCached =
        std::min((int)sortedAccesses_.size(), context_.getHardwareConfiguration().SMemMaxFields);
    for(int i = 0; i < numVarsToBeCached; ++i) {
      int oldID = sortedAccesses_[i].accessID;

      // Create new temporary field, need to figure out dimensions
      // TODO sparse_dim: Should be supported: should use same code used for checks on correct
      // dimensionality in statements.
      if(instantiation_->getIIR()->getGridType() != ast::GridType::Cartesian)
        dawn_unreachable(
            "Currently creating a new temporary field is not supported for unstructured grids.");
      sir::FieldDimensions fieldDims{sir::HorizontalFieldDimension(ast::cartesian, {true, true}),
                                     true};
      // Register the new temporary in the metadata
      int newID = metadata_.insertAccessOfType(iir::FieldAccessType::StencilTemporary,
                                               "__tmp_cache_" + std::to_string(i));
      metadata_.setFieldDimensions(newID, std::move(fieldDims));

      // Rename all the fields in this multistage
      renameAccessIDInMultiStage(multiStagePrt_.get(), oldID, newID);

      oldAccessIDtoNewAccessID_.emplace(oldID, newID);
      iir::Cache& cache =
          multiStagePrt_->setCache(iir::Cache::CacheType::IJ, iir::Cache::IOPolicy::local, newID);
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
    std::unique_ptr<iir::Stage> assignmentStage = std::make_unique<iir::Stage>(
        instantiation_->getMetaData(), instantiation_->nextUID(), interval);
    iir::Stage::DoMethodSmartPtr_t doMethod =
        std::make_unique<iir::DoMethod>(interval, instantiation_->getMetaData());
    doMethod->getAST().clear();

    for(int i = 0; i < assignmentIDs.size(); ++i) {
      int assignmentID = assignmentIDs[i];
      int assigneeID = assigneeIDs[i];
      addAssignmentToDoMethod(doMethod, assignmentID, assigneeID);
    }

    // Add the single do method to the new Stage
    assignmentStage->clearChildren();
    assignmentStage->addDoMethod(std::move(doMethod));
    for(auto& doMethod : assignmentStage->getChildren()) {
      doMethod->update(iir::NodeUpdateType::level);
    }
    assignmentStage->update(iir::NodeUpdateType::level);

    return assignmentStage;
  }

  ///@brief Add the assignment operator of two unique id's to a given doMethod
  void addAssignmentToDoMethod(const iir::Stage::DoMethodSmartPtr_t& doMethod, int assignmentID,
                               int assigneeID) {
    // Create the statement of the assignment with the new and old variables
    auto fa_assignee =
        std::make_shared<iir::FieldAccessExpr>(metadata_.getFieldNameFromAccessID(assigneeID));
    auto fa_assignment =
        std::make_shared<iir::FieldAccessExpr>(metadata_.getFieldNameFromAccessID(assignmentID));
    auto assignmentExpression =
        std::make_shared<iir::AssignmentExpr>(fa_assignment, fa_assignee, "=");
    auto expAssignment = iir::makeExprStmt(assignmentExpression);
    iir::Accesses newAccess;
    newAccess.addWriteExtent(assignmentID, iir::Extents{});
    newAccess.addReadExtent(assigneeID, iir::Extents{});
    expAssignment->getData<iir::IIRStmtData>().CallerAccesses =
        std::make_optional(std::move(newAccess));
    doMethod->getAST().push_back(std::move(expAssignment));

    // Add access ids to the expressions
    fa_assignment->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(assignmentID);
    fa_assignee->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(assigneeID);
  }

  /// @brief Checks if there is a read operation before the first write operation in the given
  /// multistage.
  /// @param[in]    AccessID  original FieldAccessID of the field to check
  /// @return true if there is a read-access before the first write access
  bool checkReadBeforeWrite(int AccessID) {

    for(const auto& stmt : iterateIIROverStmt(*multiStagePrt_)) {

      const auto& callerAccesses = stmt->getData<iir::IIRStmtData>().CallerAccesses;

      // Find first if this statement has a read
      auto readAccessIterator = callerAccesses->getReadAccesses().find(AccessID);
      if(readAccessIterator != callerAccesses->getReadAccesses().end()) {
        return true;
      }
      // If we did not find a read statement so far, we have  a write first and do not need to
      // fill the cache
      auto writeAccessIterator = callerAccesses->getWriteAccesses().find(AccessID);
      if(writeAccessIterator != callerAccesses->getWriteAccesses().end()) {
        return false;
      }
    }
    return false;
  }

  bool checkReadOnlyAccess(int AccessID) {

    for(const auto& stmt : iterateIIROverStmt(*multiStagePrt_)) {

      const auto& callerAccesses = stmt->getData<iir::IIRStmtData>().CallerAccesses;

      // If we find a write-statement we exit
      auto writeAccessIterator = callerAccesses->getWriteAccesses().find(AccessID);
      if(writeAccessIterator != callerAccesses->getWriteAccesses().end()) {
        return false;
      }
    }
    return true;
  }
};

PassSetNonTempCaches::PassSetNonTempCaches(OptimizerContext& context)
    : Pass(context, "PassSetNonTempCaches") {}

// TODO delete this pass
bool dawn::PassSetNonTempCaches::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    const iir::Stencil& stencil = *stencilPtr;

    std::vector<NameToImprovementMetric> allCachedFields;
    if(context_.getOptions().UseNonTempCaches) {
      for(const auto& multiStagePtr : stencil.getChildren()) {
        GlobalFieldCacher organizer(multiStagePtr, stencilInstantiation, context_);
        organizer.process();
        if(context_.getOptions().ReportPassSetNonTempCaches) {
          for(const auto& nametoCache : organizer.getOriginalNameToCache())
            allCachedFields.push_back(nametoCache);
        }
      }
    }
    // Output
    if(context_.getOptions().ReportPassSetNonTempCaches) {
      std::sort(allCachedFields.begin(), allCachedFields.end(),
                [](const NameToImprovementMetric& lhs, const NameToImprovementMetric& rhs) {
                  return lhs.name < rhs.name;
                });
      std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName() << " :";
      for(const auto& nametoCache : allCachedFields) {
        std::cout << " Cached: " << nametoCache.name
                  << " : Type: " << nametoCache.cache.getTypeAsString() << ":"
                  << nametoCache.cache.getIOPolicyAsString();
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
