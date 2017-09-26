//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_OPTIMIZER_DOMETHOD_H
#define GSL_OPTIMIZER_DOMETHOD_H

#include "gsl/Optimizer/Interval.h"
#include <memory>
#include <vector>

namespace gsl {

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
};

} // namespace gsl

#endif
