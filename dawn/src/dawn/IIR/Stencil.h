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

#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/SIR/SIR.h"
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

namespace dawn {

namespace iir {

class IIR;
class StencilMetaInformation;

/// @brief A Stencil is represented by a collection of MultiStages
/// @ingroup optimizer
class Stencil : public IIRNode<IIR, Stencil, MultiStage, impl::StdList> {
  const StencilMetaInformation& metadata_;
  sir::Attr stencilAttributes_;

  /// Identifier of the stencil. Note that this ID is only for code-generation to associate the
  /// stencil with a stencil-call in the run() method
  int StencilID_;

public:
  // FieldInfo desribes the properties of a given Field
  struct FieldInfo {
    FieldInfo(bool t, std::string fieldName, dawn::sir::FieldDimensions dim, const Field& f)
        : Name(fieldName), field(f), IsTemporary(t) {}

    std::string Name;
    Field field;
    bool IsTemporary;
    json::json jsonDump() const;

    bool operator==(const FieldInfo& other) const {
      return Name == other.Name && field == other.field && IsTemporary == other.IsTemporary;
    }
  };

private:
  struct DerivedInfo {
    /// Dependency graph of the stages of this stencil
    std::optional<DependencyGraphStage> stageDependencyGraph_;
    /// field info properties
    std::unordered_map<int, FieldInfo> fields_;

    void clear();
  };

  DerivedInfo derivedInfo_;

public:
  static constexpr const char* name = "Stencil";

  using MultiStageSmartPtr_t = child_smartptr_t<MultiStage>;

  /// @brief Position of a stage
  ///
  /// The position first identifies the multi-stage (`MultiStageIndex`) and afterwards the stage
  /// within the multi-stage (`StageOffset`). A `StageOffset` of -1 means @b before the first stage
  /// in the multi-stage.
  ///
  /// @b Example:
  /** @verbatim

      +- MultiStage0-+
      |              | <------ Position(0, -1)
      | +----------+ |
      | |  Stage0  | | <-----  Position(0, 0)
      | +----------+ |
      |              |
      | +----------+ |
      | |  Stage1  | | <-----  Position(0, 1)
      | +----------+ |
      +--------------+

      +- MultiStage1-+
      |              | <------ Position(1, -1)
      | +----------+ |
      | |  Stage0  | | <------ Position(1, 0)
      | +----------+ |
      +--------------+

      @endverbatim
  */
  struct StagePosition {
    StagePosition(int multiStageIndex, int stageOffset)
        : MultiStageIndex(multiStageIndex), StageOffset(stageOffset) {}

    StagePosition() : MultiStageIndex(-1), StageOffset(-1) {}
    StagePosition(const StagePosition&) = default;
    StagePosition(StagePosition&&) = default;
    StagePosition& operator=(const StagePosition&) = default;
    StagePosition& operator=(StagePosition&&) = default;

    bool operator<(const StagePosition& other) const;
    bool operator==(const StagePosition& other) const;
    bool operator!=(const StagePosition& other) const;

    /// Index of the Multi-Stage
    int MultiStageIndex;

    /// Index of the Stage inside the Multi-Stage, -1 indicates one before the first
    int StageOffset;
  };

  /// @brief Position of a statement inside a stage
  struct StatementPosition {
    StatementPosition(StagePosition stagePos, int doMethodIndex, int statementIndex)
        : StagePos(stagePos), DoMethodIndex(doMethodIndex), StatementIndex(statementIndex) {}

    StatementPosition() : StagePos(), DoMethodIndex(-1), StatementIndex(-1) {}
    StatementPosition(const StatementPosition&) = default;
    StatementPosition(StatementPosition&&) = default;
    StatementPosition& operator=(const StatementPosition&) = default;
    StatementPosition& operator=(StatementPosition&&) = default;

    bool operator<(const StatementPosition& other) const;
    bool operator<=(const StatementPosition& other) const;
    bool operator==(const StatementPosition& other) const;
    bool operator!=(const StatementPosition& other) const;

    /// @brief Check if `other` is in the same Do-Method as `this`
    bool inSameDoMethod(const StatementPosition& other) const;

    /// Position of the Stage
    StagePosition StagePos;

    /// Index of the Do-Method inside the Stage, -1 indicates one before the first
    int DoMethodIndex;

    /// Index of the Statement inside the Do-Method, -1 indicates one before the first
    int StatementIndex;
  };

  /// @brief Lifetime of a field or variable, given as an interval of `StatementPosition`s
  ///
  /// The field lifes between [Begin, End].
  struct Lifetime {
    Lifetime(StatementPosition begin, StatementPosition end) : Begin(begin), End(end) {}
    Lifetime() = default;
    Lifetime(const Lifetime&) = default;
    Lifetime(Lifetime&&) = default;
    Lifetime& operator=(const Lifetime&) = default;
    Lifetime& operator=(Lifetime&&) = default;

    StatementPosition Begin;
    StatementPosition End;

    /// @brief Check if `this` overlaps with `other`
    bool overlaps(const Lifetime& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Lifetime& lifetime);
  };

  /// @name Constructors and Assignment
  /// @{
  Stencil(const StencilMetaInformation& metadata, sir::Attr attributes, int StencilID);

  Stencil(Stencil&&) = default;
  /// @}

  /// @brief clone the stencil returning a smart ptr
  std::unique_ptr<Stencil> clone() const;

  /// @brief return the meta information
  const StencilMetaInformation& getMetadata() const { return metadata_; }

  /// @brief Compute a set of intervals for this stencil
  std::unordered_set<Interval> getIntervals() const;

  /// @brief Get the global variables referenced by this stencil
  std::vector<std::string> getGlobalVariables() const;

  /// @brief returns true if the stencils uses global variables
  bool hasGlobalVariables() const;

  /// @brief returns true if the accessid is used within the stencil
  bool hasFieldAccessID(const int accessID) const { return derivedInfo_.fields_.count(accessID); }

  /// @brief Get the enclosing interval of accesses of temporaries used in this stencil
  std::optional<Interval> getEnclosingIntervalTemporaries() const;

  /// @brief Get the multi-stage at given multistage index
  const std::unique_ptr<MultiStage>& getMultiStageFromMultiStageIndex(int multiStageIdx) const;

  /// @brief Get the multi-stage at given stage index
  const std::unique_ptr<MultiStage>& getMultiStageFromStageIndex(int stageIdx) const;

  /// @brief Get the position of the stage which is identified by the linear stage index
  StagePosition getPositionFromStageIndex(int stageIdx) const;
  int getStageIndexFromPosition(const StagePosition& position) const;

  /// @brief Get the stage at given linear stage index or position
  /// @{
  const std::unique_ptr<Stage>& getStage(int stageIdx) const;
  const std::unique_ptr<Stage>& getStage(const StagePosition& position) const;
  /// @}

  /// @brief Get the unique `StencilID`
  int getStencilID() const { return StencilID_; }

  /// @brief Insert the `stage` @b after the given `position`
  void insertStage(const StagePosition& position, std::unique_ptr<Stage>&& stage);

  /// @brief Get number of stages
  int getNumStages() const;

  json::json jsonDump() const;

  /// @brief Run `func` on each Stmt of the stencil (or on the given
  /// Stmt of the stages specified in `lifetime`)
  ///
  /// @param func           Function to run on all statements of each Do-Method
  /// @param updateFields   Update the fields afterwards
  /// @{
  void forEachStatement(std::function<void(ArrayRef<std::shared_ptr<iir::Stmt>>)> func,
                        bool updateFields = false);
  void forEachStatement(std::function<void(ArrayRef<std::shared_ptr<iir::Stmt>>)> func,
                        const Lifetime& lifetime, bool updateFields = false);
  /// @}

  /// @brief Update the fields of the stages in the stencil (or the stages specified in `lifetime`)
  /// @{
  void updateFields();
  void updateFields(const Lifetime& lifetime);
  /// @}

  /// @brief Get/Set the dependency graph of the stages
  /// @{
  const std::optional<DependencyGraphStage>& getStageDependencyGraph() const;
  void setStageDependencyGraph(DependencyGraphStage&& stageDAG);
  /// @}

  /// @brief determines whether the stencil contains redundant computations, i.e. if any of the
  /// stages has a non null extent
  bool containsRedundantComputations() const;

  /// @brief Get the axis of the stencil (i.e the interval of all stages)
  ///
  /// @param useExtendedInterval    Merge the extended intervals
  Interval getAxis(bool useExtendedInterval = true) const;

  /// @brief clear the derived info
  virtual void clearDerivedInfo() override;

  /// @brief Compute the life-time of the fields (or variables) given as a set of `AccessID`s
  std::unordered_map<int, Lifetime> getLifetime(const std::unordered_set<int>& AccessID) const;

  /// @brief get the lifetime where an access id is used
  Lifetime getLifetime(const int AccessIDs) const;

  /// @brief Check if the stencil is empty (i.e contains no statements)
  bool isEmpty() const;

  /// @brief Get the SIR Stencil
  const std::shared_ptr<sir::Stencil> getSIRStencil() const;

  /// @brief Apply the visitor to all statements in the stencil
  void accept(iir::ASTVisitor& visitor);

  /// @brief Get the pair <AccessID, field> for the fields used within the multi-stage
  const std::unordered_map<int, FieldInfo>& getFields() const { return derivedInfo_.fields_; }

  /// @brief Get the pair <AccessID, field> for the fields used within the multi-stage
  std::map<int, FieldInfo> getOrderedFields() const {
    return support::orderMap(derivedInfo_.fields_);
  }

  std::unordered_map<int, Field> computeFieldsOnTheFly() const;

  /// @brief update the derived info from children
  virtual void updateFromChildren() override;

  /// @brief compare stored derived info with the computed on the fly algorithm (in order to check
  /// that the update info is consistent with the current state of the tree)
  bool compareDerivedInfo() const;

  ///@brief Get the Attributes of the Stencil as specified in the user-code
  sir::Attr& getStencilAttributes();

private:
  void forEachStatementImpl(std::function<void(ArrayRef<std::shared_ptr<iir::Stmt>>)> func,
                            int startStageIdx, int endStageIdx, bool updateFields);
  void updateFieldsImpl(int startStageIdx, int endStageIdx);
};
} // namespace iir

} // namespace dawn
