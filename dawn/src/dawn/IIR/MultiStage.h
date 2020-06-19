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

#pragma once

#include "dawn/IIR/Cache.h"
#include "dawn/IIR/IIRNode.h"
#include "dawn/IIR/LoopOrder.h"
#include "dawn/IIR/MultiInterval.h"
#include "dawn/IIR/Stage.h"
#include <deque>
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace dawn {
class OptimizerContext;
namespace iir {

class Stencil;
class DependencyGraphAccesses;
class StencilMetaInformation;

namespace impl {
template <typename T>
using StdList = std::list<T, std::allocator<T>>;
}

/// @brief A MultiStage is represented by a collection of stages and a given exectuion policy.
///
/// A MultiStage usually corresponds to the outer loop (usually over k) of the loop nest. In CUDA
/// gridtools multistages reflect kernels.
///
/// @ingroup optimizer
class MultiStage : public IIRNode<Stencil, MultiStage, Stage, impl::StdList> {
  StencilMetaInformation& metadata_;

  LoopOrderKind loopOrder_;

  int id_;

  struct DerivedInfo {
    ///@brrief filled by PassSetCaches and PassSetNonTempCaches
    std::unordered_map<int, iir::Cache> caches_;

    std::unordered_map<int, Field> fields_;
    void clear();
  };

  DerivedInfo derivedInfo_;

public:
  static constexpr const char* name = "MultiStage";

  using StageSmartPtr_t = child_smartptr_t<Stage>;
  using ChildrenIterator = std::vector<child_smartptr_t<Stage>>::iterator;

  /// @name Constructors and Assignment
  /// @{
  MultiStage(StencilMetaInformation& metadata, LoopOrderKind loopOrder);
  MultiStage(MultiStage&&) = default;
  /// @}

  std::unique_ptr<MultiStage> clone() const;

  json::json jsonDump() const;

  /// @brief Get the loop order
  LoopOrderKind getLoopOrder() const { return loopOrder_; }

  int getID() const { return id_; }
  /// @}

  void setID(int id) { id_ = id; }

  /// @brief clear the derived info
  virtual void clearDerivedInfo() override;

  std::vector<std::unique_ptr<DoMethod>> computeOrderedDoMethods() const;

  /// @brief Set the loop order
  void setLoopOrder(LoopOrderKind loopOrder) { loopOrder_ = loopOrder; }

  virtual void updateFromChildren() override;

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
  std::vector<std::unique_ptr<MultiStage>>
  split(std::deque<MultiStage::SplitIndex>& splitterIndices, LoopOrderKind lastLoopOrder);

  /// @brief Get the dependency graph of the multi-stage incorporating those stages whose extended
  /// interval overlaps with `interval`
  DependencyGraphAccesses getDependencyGraphOfInterval(const Interval& interval) const;

  /// @brief Get the dependency graph of the multi-stage incorporating all stages
  DependencyGraphAccesses getDependencyGraphOfAxis() const;

  /// @brief Set a cache
  iir::Cache& setCache(iir::Cache::CacheType type, iir::Cache::IOPolicy policy, int AccessID,
                       Interval const& interval, const Interval& enclosingAccessedInterval,
                       std::optional<iir::Cache::window> w);

  iir::Cache& setCache(iir::Cache::CacheType type, iir::Cache::IOPolicy policy, int AccessID);

  /// @brief computes the interval where an accessId is used (extended by the extent of the
  /// access)
  std::optional<Interval> computeEnclosingAccessInterval(const int accessID,
                                                         const bool mergeWithDoInterval) const;

  /// @brief Is the field given by the `AccessID` cached?
  bool isCached(int AccessID) const { return derivedInfo_.caches_.count(AccessID); }

  /// @brief returns the last interval level that was computed by an accessID
  Interval::IntervalLevel lastLevelComputed(const int accessID) const;

  /// @brief Get the intervals of the multi-stage
  std::unordered_set<Interval> getIntervals() const;

  /// @brief Get the enclosing vertical Interval of this multi-stage
  ///
  /// This merges the intervals of all Do-Methods of all stages.
  Interval getEnclosingInterval() const;

  /// @brief Get the pair <AccessID, field> for the fields used within the multi-stage
  const std::unordered_map<int, Field>& getFields() const;
  std::map<int, Field> getOrderedFields() const;

  /// @brief Compute and return the pairs <AccessID, field> used for a given interval
  std::unordered_map<int, Field> computeFieldsAtInterval(const iir::Interval& interval) const;

  /// @brief determines whether an accessID corresponds to a temporary that will perform accesses to
  /// main memory
  bool isMemAccessTemporary(const int accessID) const;

  /// @brief true if there is at least a temporary that requires access to main mem
  bool hasMemAccessTemporaries() const;

  /// @brief determines whether the multistage contains the field with an accessID
  bool hasField(const int accessID) const;

  /// @brief field getter with an accessID
  const Field& getField(int accessID) const;

  /// @brief computes the collection of fields of the multistage on the fly (returns copy)
  std::unordered_map<int, Field> computeFieldsOnTheFly() const;

  /// @brief Get the enclosing interval of all access to temporaries
  std::optional<Interval> getEnclosingAccessIntervalTemporaries() const;

  /// @brief Get the caches
  /// //TODO remove this non const getter
  std::unordered_map<int, iir::Cache>& getCaches() { return derivedInfo_.caches_; }
  const std::unordered_map<int, iir::Cache>& getCaches() const { return derivedInfo_.caches_; }

  const iir::Cache& getCache(const int accessID) const;

  /// @brief true if it contains no stages or the stages are empty
  bool isEmptyOrNullStmt() const;

  // TODO doc
  MultiInterval computeReadAccessInterval(int accessID) const;

  /// @brief returns the vertical extent of the kcache associated with an accessID
  /// (determined from all accesses of a full MS)
  Extent getKCacheVertExtent(const int accessID) const;

  /// @brief computes the extents of an accessID field at a given interval
  std::optional<Extents> computeExtents(const int accessID, const Interval& interval) const;

  MultiInterval computePartitionOfIntervals() const;

  StencilMetaInformation& getMetadata();
};

} // namespace iir
} // namespace dawn
