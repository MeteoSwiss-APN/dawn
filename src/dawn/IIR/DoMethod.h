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

#ifndef DAWN_IIR_DOMETHOD_H
#define DAWN_IIR_DOMETHOD_H

#include "dawn/IIR/IIRNode.h"
#include "dawn/Optimizer/Interval.h"
#include <boost/optional.hpp>
#include <memory>
#include <vector>

namespace dawn {
namespace iir {

class Stage;
class DependencyGraphAccesses;
class StatementAccessesPair;

/// @brief A Do-method is a collection of Statements with corresponding Accesses of a specific
/// vertical region
///
/// @ingroup optimizer
class DoMethod : public IIRNode<void, DoMethod, StatementAccessesPair> {
  Interval interval_;
  const long unsigned int id_;

  std::shared_ptr<DependencyGraphAccesses> dependencyGraph_;
  //  std::vector<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs_;

public:
  using StatementAccessesIterator = child_iterator_t;

  /// @name Constructors and Assignment
  /// @{
  DoMethod(Interval interval);

  DoMethod(const DoMethod&) = default;
  DoMethod(DoMethod&&) = default;

  DoMethod& operator=(const DoMethod&) = default;
  DoMethod& operator=(DoMethod&&) = default;
  /// @}

  //  /// @brief Get the statements of the Stage
  //  std::vector<std::shared_ptr<StatementAccessesPair>>& getStatementAccessesPairs();
  //  const std::vector<std::shared_ptr<StatementAccessesPair>>& getStatementAccessesPairs() const;

  /// @brief Get the vertical Interval
  Interval& getInterval();
  const Interval& getInterval() const;

  void setInterval(Interval const& interval);

  unsigned long int getID() const { return id_; }

  /// @brief Set the dependency graph of this Do-Method
  void setDependencyGraph(const std::shared_ptr<DependencyGraphAccesses>& DG);

  /// @brief Get the dependency graph of this Do-Method
  std::shared_ptr<DependencyGraphAccesses>& getDependencyGraph();
  const std::shared_ptr<DependencyGraphAccesses>& getDependencyGraph() const;

  /// @brief computes the maximum extent among all the accesses of accessID
  boost::optional<Extents> computeMaximumExtents(const int accessID) const;

  /// @brief true if it is empty
  bool isEmptyOrNullStmt() const;

  /// @param accessID accessID for which the enclosing interval is computed
  /// @param mergeWidhDoInterval determines if the extent of the access is merged with the interval
  /// of the do method.
  /// Example:
  ///    do(kstart+2,kend) return u[k+1]
  /// will return Interval{3,kend+1} if mergeWithDoInterval is false
  /// will return Interval{2,kend+1} (which is Interval{3,kend+1}.merge(Interval{2,kend})) if
  /// mergeWithDoInterval is true
  boost::optional<Interval> computeEnclosingAccessInterval(const int accessID,
                                                           const bool mergeWithDoInterval) const;
};

} // namespace iir
} // namespace dawn

#endif
