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

#ifndef DAWN_OPTIMIZER_DOMETHOD_H
#define DAWN_OPTIMIZER_DOMETHOD_H

#include "dawn/Optimizer/Interval.h"
#include <boost/optional.hpp>
#include <memory>
#include <vector>

namespace dawn {

class Stage;
class DependencyGraphAccesses;
class StatementAccessesPair;

/// @brief A Do-method is a collection of Statements with corresponding Accesses of a specific
/// vertical region
///
/// @ingroup optimizer
class DoMethod {
  Stage* stage_;
  Interval interval_;

  std::shared_ptr<DependencyGraphAccesses> dependencyGraph_;
  std::vector<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs_;

public:
  /// @name Constructors and Assignment
  /// @{
  DoMethod(Stage* stage, Interval interval);

  DoMethod(const DoMethod&) = default;
  DoMethod(DoMethod&&) = default;

  DoMethod& operator=(const DoMethod&) = default;
  DoMethod& operator=(DoMethod&&) = default;
  /// @}

  /// @brief Get the statements of the Stage
  std::vector<std::shared_ptr<StatementAccessesPair>>& getStatementAccessesPairs();
  const std::vector<std::shared_ptr<StatementAccessesPair>>& getStatementAccessesPairs() const;

  /// @brief Get the vertical Interval
  Interval& getInterval();
  const Interval& getInterval() const;

  /// @brief Get the associated `stage`
  Stage* getStage();

  /// @brief Set the dependency graph of this Do-Method
  void setDependencyGraph(const std::shared_ptr<DependencyGraphAccesses>& DG);

  /// @brief Get the dependency graph of this Do-Method
  std::shared_ptr<DependencyGraphAccesses>& getDependencyGraph();
  const std::shared_ptr<DependencyGraphAccesses>& getDependencyGraph() const;

  /// @brief computes the maximum extent among all the accesses of accessID
  boost::optional<Extents> computeMaximumExtents(const int accessID) const;

  /// @brief computes the interval where an accessId is used (extended by the extent of the
  /// access)
  boost::optional<Interval> computeEnclosingAccessInterval(const int accessID) const;
};

} // namespace dawn

#endif
