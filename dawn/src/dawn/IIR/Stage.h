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

#ifndef DAWN_IIR_STAGE_H
#define DAWN_IIR_STAGE_H

#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/Field.h"
#include "dawn/IIR/IIRNode.h"
#include "dawn/IIR/Interval.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Support/ArrayRef.h"
#include "dawn/Support/ContainerUtils.h"
#include <deque>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

namespace dawn {
namespace iir {

class DependencyGraphAccesses;
class MultiStage;

/// @brief A Stage is represented by a collection of statements grouped into DoMethod of
/// non-overlapping vertical intervals.
///
/// A Stage usually corresponds to a inner loop nest (usually an ij-loop). In CUDA gridtools stages
/// are separated by a `__syncthreads()` call in a kernel.
///
/// @ingroup optimizer
class Stage : public IIRNode<MultiStage, Stage, DoMethod> {

  const StencilMetaInformation& metaData_;

  /// Unique identifier of the stage
  int StageID_;

  struct DerivedInfo {

    DerivedInfo() : extents_{0, 0, 0, 0, 0, 0} {}
    DerivedInfo(DerivedInfo&&) = default;
    DerivedInfo(const DerivedInfo&) = default;
    DerivedInfo& operator=(DerivedInfo&&) = default;
    DerivedInfo& operator=(const DerivedInfo&) = default;

    void clear();

    /// Declaration of the fields of this stage
    std::unordered_map<int, Field> fields_;

    /// AccessIDs of the global variable accesses of this stage
    std::unordered_set<int> allGlobalVariables_;
    std::unordered_set<int> globalVariables_;
    std::unordered_set<int> globalVariablesFromStencilFunctionCalls_;

    Extents extents_;
    bool requiresSync_ = false;
  };

  DerivedInfo derivedInfo_;

public:
  static constexpr const char* name = "Stage";

  using DoMethodSmartPtr_t = child_smartptr_t<DoMethod>;
  using ChildrenIterator = std::vector<child_smartptr_t<DoMethod>>::iterator;

  /// @name Constructors and Assignment
  /// @{
  Stage(const StencilMetaInformation& metaData, int StageID, const Interval& interval);
  Stage(const StencilMetaInformation& metaData, int StageID);

  Stage(Stage&&) = default;
  /// @}

  std::unique_ptr<Stage> clone() const;

  json::json jsonDump(const StencilMetaInformation& metaData) const;

  /// @brief update the derived info from children
  virtual void updateFromChildren() override;

  /// @brief update the global variables derived info
  void updateGlobalVariablesInfo();

  /// @brief clear the derived info
  virtual void clearDerivedInfo() override;

  /// @brief Check if the stage contains of a single Do-Method
  bool hasSingleDoMethod() const;

  /// @brief Get the single Do-Method
  DoMethod& getSingleDoMethod();
  const DoMethod& getSingleDoMethod() const;

  /// @brief Get the unique `StageID`
  int getStageID() const { return StageID_; }

  /// @brief Get the vertical Interval of this stage
  std::vector<Interval> getIntervals() const;

  /// @brief Get the enclosing vertical Interval of this stage
  ///
  /// This merges the intervals of all Do-Methods.
  Interval getEnclosingInterval() const;

  /// @brief Get the extended enclosing vertical Interval incorporating vertical extents of the
  /// fields
  ///
  /// @see getEnclosingInterval
  Interval getEnclosingExtendedInterval() const;

  /// @brief Check if the fields of the `other` stage overlap with the fields of this stage in the
  /// vertical interval (incorporating the vertical extents of the fields)
  ///
  /// Note that this is more accurate than
  /// @code
  ///   bool stagesOverlap = this->getExtendedInterval().overlaps(other.getExtendedInterval());
  /// @endcode
  /// as the check is performed on each individual field.
  ///
  /// @{
  bool overlaps(const Stage& other) const;
  bool overlaps(const Interval& interval, const std::unordered_map<int, Field>& fields) const;
  /// @}

  /// @brief Get the maximal vertical extent of this stage
  Extent getMaxVerticalExtent() const;

  /// @brief computes the interval where an accessId is used (extended by the extent of the access)
  std::optional<Interval> computeEnclosingAccessInterval(const int accessID,
                                                         const bool mergeWithDoInterval) const;

  /// @brief Get fields of this stage sorted according their Intend: `Output` -> `IntputOutput` ->
  /// `Input`
  ///
  /// The fields are computed during `Stage::update`.
  const std::unordered_map<int, Field>& getFields() const { return derivedInfo_.fields_; }

  std::map<int, Field> getOrderedFields() const { return support::orderMap(derivedInfo_.fields_); }

  /// @brief Update the fields and global variables
  ///
  /// This recomputes the fields referenced in this Stage (and all its Do-Methods) and computes
  /// the @b accumulated extent of each field
  virtual void updateLevel() override;

  /// @brief checks whether the stage contains global variables
  bool hasGlobalVariables() const;

  /// @brief Get the global variables accessed in this stage
  ///
  /// only returns those global variables used within the stage,
  /// but not inside stencil functions called from the stage
  /// The global variables are computed during `Stage::update`.
  const std::unordered_set<int>& getGlobalVariables() const;

  /// @brief Get the global variables accessed in stencil functions that
  /// are called from within the stage
  ///
  /// The global variables are computed during `Stage::update`.
  const std::unordered_set<int>& getGlobalVariablesFromStencilFunctionCalls() const;

  /// @brief Get the all global variables used in the stage:
  /// i.e. the union of getGlovalVariables() and getGlobalVariablesFromStencilFunctionCalls()
  ///
  /// The global variables are computed during `Stage::update`.
  const std::unordered_set<int>& getAllGlobalVariables() const;

  /// @brief Add the given Do-Method to the list of Do-Methods of this stage
  ///
  /// Calls `update()` in the end.
  void addDoMethod(const DoMethodSmartPtr_t& doMethod);

  /// @brief Append the `from` DoMethod to the existing `to` DoMethod of this stage and use
  /// `dependencyGraph` as the new DependencyGraphAccesses of this new Do-Method
  ///
  /// Calls `update()` in the end.
  void appendDoMethod(DoMethodSmartPtr_t& from, DoMethodSmartPtr_t& to,
                      const std::shared_ptr<DependencyGraphAccesses>& dependencyGraph);

  /// @brief Split the stage at the given indices into separate stages.
  ///
  /// This stage will not be usable after this operations i.e it's members will be moved into the
  /// new stages. This function consumes the input argument `splitterIndices`.
  ///
  /// If a vector of graphs is provided, it will be assigned to the new stages.
  ///
  /// @return New stages
  std::vector<std::unique_ptr<Stage>>
  split(std::deque<int>& splitterIndices,
        const std::deque<std::shared_ptr<DependencyGraphAccesses>>* graphs);

  /// @brief Get the extent of the stage
  /// @{
  Extents const& getExtents() const { return derivedInfo_.extents_; }
  void setExtents(Extents const& extents) { derivedInfo_.extents_ = extents; }
  /// @}

  /// @brief true if it contains no do methods or they are empty
  bool isEmptyOrNullStmt() const;

  /// @brief set the flag that specifies that the stage will require an explicit sync before
  /// execution
  void setRequiresSync(const bool sync);
  /// @brief get the flag that specifies that the stage will require an explicit sync before
  /// execution
  bool getRequiresSync() const;
};

} // namespace iir
} // namespace dawn

#endif
