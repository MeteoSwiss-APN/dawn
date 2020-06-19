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

#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/Field.h"
#include "dawn/IIR/IIRNode.h"
#include "dawn/IIR/Interval.h"
#include "dawn/IIR/StencilMetaInformation.h"
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

  // Location type of the stage (which loop it represents)
  std::optional<ast::LocationType> type_;

  /// Iteration space in the horizontal. If it is not instantiated, iteration over the full domain
  /// is assumed. This is built on top of the DerivedInfo::Extents class and for a full compute
  /// domain, those have to be added together.
  std::array<std::optional<Interval>, 2> iterationSpace_;

  struct DerivedInfo {

    void clear();

    /// Declaration of the fields of this stage
    std::unordered_map<int, Field> fields_;

    /// AccessIDs of the global variable accesses of this stage
    std::unordered_set<int> allGlobalVariables_;
    std::unordered_set<int> globalVariables_;
    std::unordered_set<int> globalVariablesFromStencilFunctionCalls_;

    // Further data (not cleared!)
    Extents extents_;           // valid after StencilInstantiation::computeDerivedInfo
    bool requiresSync_ = false; // valid after PassSetSyncStage
  };

  DerivedInfo derivedInfo_;

public:
  static constexpr const char* name = "Stage";

  using IterationSpace = std::array<std::optional<Interval>, 2>;
  using DoMethodSmartPtr_t = child_smartptr_t<DoMethod>;
  using ChildrenIterator = std::vector<child_smartptr_t<DoMethod>>::iterator;

  /// @name Constructors and Assignment
  /// @{
  Stage(const StencilMetaInformation& metaData, int StageID,
        IterationSpace iterationspace = {std::optional<Interval>(), std::optional<Interval>()});

  Stage(const StencilMetaInformation& metaData, int StageID, const Interval& interval,
        IterationSpace iterationspace = {std::optional<Interval>(), std::optional<Interval>()});

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

  /// @brief Moves the given Do-Method to the list of Do-Methods of this stage
  ///
  /// Calls `update()` in the end.
  void addDoMethod(DoMethodSmartPtr_t&& doMethod);

  /// @brief Append the `from` DoMethod to the existing `to` DoMethod of this stage and use
  /// `dependencyGraph` as the new DependencyGraphAccesses of this new Do-Method
  ///
  /// Calls `update()` in the end.
  void appendDoMethod(DoMethodSmartPtr_t& from, DoMethodSmartPtr_t& to,
                      DependencyGraphAccesses&& dependencyGraph);

  /// @brief Split the stage at the given indices into separate stages.
  ///
  /// This stage will not be usable after this operations i.e it's members will be moved into the
  /// new stages. This function consumes the input argument `splitterIndices`.
  ///
  /// @return New stages
  std::vector<std::unique_ptr<Stage>> split(std::deque<int> const& splitterIndices);

  /// @brief Split the stage at the given indices into separate stages and updates the stages with
  /// graph.
  ///
  /// @return New stages
  std::vector<std::unique_ptr<Stage>> split(std::deque<int> const& splitterIndices,
                                            std::deque<DependencyGraphAccesses>&& graphs);

  /// @brief Get the extent of the stage
  /// @{
  Extents const& getExtents() const { return derivedInfo_.extents_; }
  void setExtents(Extents const& extents) { derivedInfo_.extents_ = extents; }
  /// @}

  /// @brief true if it contains no do methods or they are empty
  bool isEmptyOrNullStmt() const;

  /// @brief Get the flag that specifies that the space will require an explicit sync before
  /// execution
  /// @{
  void setRequiresSync(const bool sync);
  bool getRequiresSync() const;
  /// @}

  /// @brief setter for the location type
  void setLocationType(ast::LocationType type);

  /// @brief returns the location type of a stage
  std::optional<ast::LocationType> getLocationType() const;
  /// @brief Get the horizontal iteration space
  /// @{
  void setIterationSpace(const IterationSpace& value);
  const IterationSpace& getIterationSpace() const;
  bool hasIterationSpace() const;
  /// @}

  /// @brief Are iteration spaces of this stage and another stage compatible?
  ///
  /// Iteration spaces are said to be compatible if they are either equal, or the iteration
  /// space of the `other` stage is completely contained in this stages iteration space
  /// If this stage contains an iteration space and the other does not, the iteration spaces are
  /// incompatible If neither stage or just the other stage contains an iteration space they are
  /// compatible
  bool iterationSpaceCompatible(const Stage& other) const;
};

} // namespace iir
} // namespace dawn
