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

#ifndef DAWN_OPTIMIZER_STENCIL_H
#define DAWN_OPTIMIZER_STENCIL_H

#include "dawn/Optimizer/MultiStage.h"
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/Statement.h"
#include <functional>
#include <list>
#include <memory>
#include <unordered_set>
#include <vector>

namespace dawn {

class StencilInstantiation;
class DependencyGraphStage;
class StatementAccessesPair;

/// @brief A Stencil is represented by a collection of MultiStages
/// @ingroup optimizer
class Stencil {
  StencilInstantiation& stencilInstantiation_;
  const std::shared_ptr<sir::Stencil> SIRStencil_;

  /// Identifier of the stencil. Note that this ID is only for code-generation to associate the
  /// stencil with a stencil-call in the run() method
  int StencilID_;

  /// Dependency graph of the stages of this stencil
  std::shared_ptr<DependencyGraphStage> stageDependencyGraph_;

  /// List of multi-stages in the stencil
  std::list<std::shared_ptr<MultiStage>> multistages_;

public:
  struct FieldInfo {
    bool IsTemporary;
    std::string Name;
    int AccessID;
    Array3i Dimensions;
  };

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

    friend std::ostream& operator<<(std::ostream& os, const StagePosition& position);
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

    friend std::ostream& operator<<(std::ostream& os, const StatementPosition& position);
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
  Stencil(StencilInstantiation& stencilInstantiation,
          const std::shared_ptr<sir::Stencil>& SIRStencil, int StencilID,
          const std::shared_ptr<DependencyGraphStage>& stageDependencyGraph = nullptr);

  Stencil(const Stencil&) = default;
  Stencil(Stencil&&) = default;

  Stencil& operator=(const Stencil&) = default;
  Stencil& operator=(Stencil&&) = default;
  /// @}

  /// @brief Compute a set of intervals for this stencil
  std::unordered_set<Interval> getIntervals() const;

  /// @brief Get the fields referenced by this stencil (temporary fields are listed first if
  /// requested)
  std::vector<FieldInfo> getFields(bool withTemporaries = true) const;

  /// @brief Get the global variables referenced by this stencil
  std::vector<std::string> getGlobalVariables() const;

  /// @brief Get the stencil instantiation
  StencilInstantiation& getStencilInstantiation() const { return stencilInstantiation_; }

  /// @brief Get the multi-stages of the stencil
  std::list<std::shared_ptr<MultiStage>>& getMultiStages() { return multistages_; }
  const std::list<std::shared_ptr<MultiStage>>& getMultiStages() const { return multistages_; }

  /// @brief Get the enclosing interval of accesses of temporaries used in this stencil
  std::shared_ptr<Interval> getEnclosingIntervalTemporaries() const;

  /// @brief Get the multi-stage at given multistage index
  const std::shared_ptr<MultiStage>& getMultiStageFromMultiStageIndex(int multiStageIdx) const;

  /// @brief Get the multi-stage at given stage index
  const std::shared_ptr<MultiStage>& getMultiStageFromStageIndex(int stageIdx) const;

  /// @brief Get the position of the stage which is identified by the linear stage index
  StagePosition getPositionFromStageIndex(int stageIdx) const;
  int getStageIndexFromPosition(const StagePosition& position) const;

  /// @brief Get the stage at given linear stage index or position
  /// @{
  const std::shared_ptr<Stage>& getStage(int stageIdx) const;
  const std::shared_ptr<Stage>& getStage(const StagePosition& position) const;
  /// @}

  /// @brief Get the unique `StencilID`
  int getStencilID() const { return StencilID_; }

  /// @brief Insert the `stage` @b after the given `position`
  void insertStage(const StagePosition& position, const std::shared_ptr<Stage>& stage);

  /// @brief Get number of stages
  int getNumStages() const;

  /// @brief Run `func` on each StatementAccessesPair of the stencil (or on the given
  /// StatementAccessesPair of the stages specified in `lifetime`)
  ///
  /// @param func           Function to run on all statements of each Do-Method
  /// @param updateFields   Update the fields afterwards
  /// @{
  void forEachStatementAccessesPair(
      std::function<void(ArrayRef<std::shared_ptr<StatementAccessesPair>>)> func,
      bool updateFields = false);
  void forEachStatementAccessesPair(
      std::function<void(ArrayRef<std::shared_ptr<StatementAccessesPair>>)> func,
      const Lifetime& lifetime, bool updateFields = false);
  /// @}

  /// @brief Update the fields of the stages in the stencil (or the stages specified in `lifetime`)
  /// @{
  void updateFields();
  void updateFields(const Lifetime& lifetime);
  /// @}

  /// @brief Get/Set the dependency graph of the stages
  /// @{
  const std::shared_ptr<DependencyGraphStage>& getStageDependencyGraph() const;
  void setStageDependencyGraph(const std::shared_ptr<DependencyGraphStage>& stageDAG);
  /// @}

  /// @brief Get the axis of the stencil (i.e the interval of all stages)
  ///
  /// @param useExtendedInterval    Merge the extended intervals
  Interval getAxis(bool useExtendedInterval = true) const;

  /// @brief Rename all occurences of field `oldAccessID` to `newAccessID`
  void renameAllOccurrences(int oldAccessID, int newAccessID);

  /// @brief Compute the life-time of the fields (or variables) given as a set of `AccessID`s
  std::unordered_map<int, Lifetime> getLifetime(const std::unordered_set<int>& AccessID) const;

  /// @brief Check if the stencil is empty (i.e contains no statements)
  bool isEmpty() const;

  /// @brief Get the SIR Stencil
  const std::shared_ptr<sir::Stencil> getSIRStencil() const;

  /// @brief Apply the visitor to all statements in the stencil
  void accept(ASTVisitor& visitor);

  /// @brief Convert stencil to string (i.e print the list of multi-stage -> stages)
  friend std::ostream& operator<<(std::ostream& os, const Stencil& stencil);

private:
  void forEachStatementAccessesPairImpl(
      std::function<void(ArrayRef<std::shared_ptr<StatementAccessesPair>>)> func, int startStageIdx,
      int endStageIdx, bool updateFields);
  void updateFieldsImpl(int startStageIdx, int endStageIdx);
};

} // namespace dawn

#endif
