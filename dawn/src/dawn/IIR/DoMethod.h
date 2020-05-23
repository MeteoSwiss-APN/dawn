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

#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/Field.h"
#include "dawn/IIR/IIRNode.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/Interval.h"
#include <memory>
#include <optional>
#include <vector>

namespace dawn {
namespace iir {

class Stage;
class StencilMetaInformation;

/// @brief A Do-method is a collection of Statements with corresponding Accesses of a specific
/// vertical region
///
/// @ingroup optimizer
class DoMethod : public IIRNode<Stage, DoMethod, void> {
  Interval interval_;
  long unsigned int id_;

  struct DerivedInfo {
    DerivedInfo clone() const;

    void clear();

    /// Declaration of the fields of this doMethod
    std::unordered_map<int, Field> fields_;
    std::optional<DependencyGraphAccesses> dependencyGraph_;
  };

  const StencilMetaInformation& metaData_;
  DerivedInfo derivedInfo_;
  std::shared_ptr<iir::BlockStmt> ast_;

public:
  static constexpr const char* name = "DoMethod";

  /// @name Constructors and Assignment
  /// @{
  DoMethod(Interval interval, const StencilMetaInformation& metaData);
  DoMethod(DoMethod&&) = default;
  /// @}

  json::json jsonDump(const StencilMetaInformation& metaData) const;

  /// @brief clone the object creating and returning a new unique_ptr
  std::unique_ptr<DoMethod> clone() const;

  /// @name Getters
  /// @{
  Interval& getInterval();
  const Interval& getInterval() const;
  inline unsigned long int getID() const { return id_; }
  const std::optional<DependencyGraphAccesses>& getDependencyGraph() const;
  /// @}

  /// @name Setters
  /// @{
  void setInterval(Interval const& interval);
  void setID(const long unsigned int id) { id_ = id; }
  void setDependencyGraph(DependencyGraphAccesses&& DG);
  /// @}

  virtual void clearDerivedInfo() override;

  /// @brief computes the maximum extent among all the accesses of accessID
  std::optional<Extents> computeMaximumExtents(const int accessID) const;

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
  std::optional<Interval> computeEnclosingAccessInterval(const int accessID,
                                                         const bool mergeWithDoInterval) const;

  /// @brief Get fields of this stage sorted according their Intend: `Output` -> `IntputOutput` ->
  /// `Input`
  ///
  /// The fields are computed during `DoMethod::update`.
  const std::unordered_map<int, Field>& getFields() const { return derivedInfo_.fields_; }

  /// @brief Get a map from field name to its dimensions for each field referenced in the DoMethod
  ///
  /// The fields are computed during `DoMethod::update`.
  const std::unordered_map<std::string, sir::FieldDimensions> getFieldDimensionsByName() const;

  bool hasField(int accessID) const { return derivedInfo_.fields_.count(accessID); }

  /// @brief field getter
  const Field& getField(const int accessID) const {
    DAWN_ASSERT(derivedInfo_.fields_.count(accessID));
    return derivedInfo_.fields_.at(accessID);
  }

  /// @brief Update the fields and global variables
  ///
  /// This recomputes the fields referenced in this Do-Method and computes
  /// the @b accumulated extent of each field
  virtual void updateLevel() override;

  /// @brief update the derived info from the children (currently no information are propagated,
  /// therefore the method is empty
  inline virtual void updateFromChildren() override {}

  void setAST(std::shared_ptr<BlockStmt> ast) { ast_ = ast; }
  iir::BlockStmt const& getAST() const { return *ast_; }
  iir::BlockStmt& getAST() { return *ast_; }
  std::shared_ptr<iir::BlockStmt> getASTPtr() { return ast_; }
};

} // namespace iir

template <typename RootNode>
auto iterateIIROverStmt(const RootNode& root) {
  std::vector<std::shared_ptr<iir::Stmt>> allStmts;
  for(auto& doMethod : iterateIIROver<iir::DoMethod>(root)) {
    std::copy(doMethod->getAST().getStatements().begin(), doMethod->getAST().getStatements().end(),
              std::back_inserter(allStmts));
  }
  return allStmts;
}
} // namespace dawn
