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

#ifndef DAWN_OPTIMIZER_MULTISTAGE_H
#define DAWN_OPTIMIZER_MULTISTAGE_H

#include "dawn/Optimizer/Cache.h"
#include "dawn/Optimizer/LoopOrder.h"
#include "dawn/Optimizer/Stage.h"
#include <deque>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace dawn {

class StencilInstantiation;
class DependencyGraphAccesses;
class OptimizerContext;

/// @brief A MultiStage is represented by a collection of stages and a given exectuion policy.
///
/// A MultiStage usually corresponds to the outer loop (usually over k) of the loop nest. In CUDA
/// gridtools multistages reflect kernels.
///
/// @ingroup optimizer
class MultiStage {
  StencilInstantiation& stencilInstantiation_;

  LoopOrderKind loopOrder_;
  std::list<std::shared_ptr<Stage>> stages_;

  std::unordered_map<int, Cache> caches_;

public:
  /// @name Constructors and Assignment
  /// @{
  MultiStage(StencilInstantiation& stencilInstantiation, LoopOrderKind loopOrder);
  MultiStage(const MultiStage&) = default;
  MultiStage(MultiStage&&) = default;

  MultiStage& operator=(const MultiStage&) = default;
  MultiStage& operator=(MultiStage&&) = default;
  /// @}

  /// @brief Get the multi-stages of the stencil
  std::list<std::shared_ptr<Stage>>& getStages() { return stages_; }
  const std::list<std::shared_ptr<Stage>>& getStages() const { return stages_; }

  /// @brief Get the execution policy
  StencilInstantiation& getStencilInstantiation() const { return stencilInstantiation_; }

  /// @brief Get the loop order
  LoopOrderKind getLoopOrder() const { return loopOrder_; }

  /// @brief Set the loop order
  void setLoopOrder(LoopOrderKind loopOrder) { loopOrder_ = loopOrder; }

  /// @brief Index containing the information for splitting MultiStages
  ///
  /// The multi stage will be split into a lower and upper multistage at the given `(StageIndex,
  /// StmtIndex)` pair.
  struct SplitIndex {
    int StageIndex;               ///< Stage to split
    int StmtIndex;                ///< Statement to split
    LoopOrderKind LowerLoopOrder; ///< New loop order of the @b lower multi-stage
  };

  /// @brief Split the multi-stage at the given indices into separate multi-stages.
  ///
  /// This multi-stage will not be usable after this operations i.e it's members will be moved into
  /// the new stages. This function consumes the input argument `splitterIndices`.
  ///
  /// @return New multi-stages
  std::vector<std::shared_ptr<MultiStage>>
  split(std::deque<MultiStage::SplitIndex>& splitterIndices, LoopOrderKind lastLoopOrder);

  /// @brief Get the dependency graph of the multi-stage incorporating those stages whose extended
  /// interval overlaps with `interval`
  std::shared_ptr<DependencyGraphAccesses>
  getDependencyGraphOfInterval(const Interval& interval) const;

  /// @brief Get the dependency graph of the multi-stage incorporating all stages
  std::shared_ptr<DependencyGraphAccesses> getDependencyGraphOfAxis() const;

  /// @brief Set a cache
  Cache& setCache(Cache::CacheTypeKind type, Cache::CacheIOPolicy policy, int AccessID,
                  Interval const& interval);

  Cache& setCache(Cache::CacheTypeKind type, Cache::CacheIOPolicy policy, int AccessID);

  /// @brief computes the interval where an accessId is used (extended by the extent of the access)
  boost::optional<Interval> computeEnclosingAccessInterval(const int accessID) const;

  /// @brief Is the field given by the `AccessID` cached?
  bool isCached(int AccessID) const { return caches_.count(AccessID); }

  /// @brief Check if the multi-stage is empty (i.e contains no statements)
  bool isEmpty() const { return stages_.empty(); }

  /// @brief Get the intervals of the multi-stage
  std::unordered_set<Interval> getIntervals() const;

  /// @brief Get the enclosing vertical Interval of this multi-stage
  ///
  /// This merges the intervals of all Do-Methods of all stages.
  Interval getEnclosingInterval() const;

  /// @brief Get the pair <AccessID, field> for the fields used within the multi-stage
  std::unordered_map<int, Field> getFields() const;

  /// @brief Get the enclosing interval of all access to temporaries
  std::shared_ptr<Interval> getEnclosingAccessIntervalTemporaries() const;

  /// @brief Get the caches
  std::unordered_map<int, Cache>& getCaches() { return caches_; }
  const std::unordered_map<int, Cache>& getCaches() const { return caches_; }

  /// @brief Rename all the occurances in the multi-stage
  void renameAllOccurrences(int oldAccessID, int newAccessID);
};

} // namespace dawn

#endif
