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

#include "dawn/IIR/ASTFwd.h"
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
class DependencyGraphAccesses;
class StatementAccessesPair;
class StencilMetaInformation;

/// @brief A Do-method is a collection of Statements with corresponding Accesses of a specific
/// vertical region
///
/// @ingroup optimizer
class DoMethod : public IIRNode<Stage, DoMethod, iir::Stmt> {
  Interval interval_;
  long unsigned int id_;

  struct DerivedInfo {
    DerivedInfo() : dependencyGraph_(nullptr) {}
    DerivedInfo(DerivedInfo&&) = default;
    DerivedInfo(const DerivedInfo&) = default;
    DerivedInfo& operator=(DerivedInfo&&) = default;
    DerivedInfo& operator=(const DerivedInfo&) = default;

    DerivedInfo clone() const;

    void clear();

    /// Declaration of the fields of this doMethod
    std::unordered_map<int, Field> fields_;
    std::shared_ptr<DependencyGraphAccesses> dependencyGraph_;
  };

  const StencilMetaInformation& metaData_;
  DerivedInfo derivedInfo_;
  iir::BlockStmt ast_;

public:
  static constexpr const char* name = "DoMethod";

  bool checkDoMethod() const override { return false; }

  // using StatementAccessesIterator = ChildIterator;

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
  const std::shared_ptr<DependencyGraphAccesses>& getDependencyGraph() const;
  /// @}

  /// @name Setters
  /// @{
  void setInterval(Interval const& interval);
  void setID(const long unsigned int id) { id_ = id; }
  void setDependencyGraph(const std::shared_ptr<DependencyGraphAccesses>& DG);
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

  const std::vector<std::shared_ptr<iir::Stmt>>& getChildren() const {
    return ast_.getStatements();
  }

  std::vector<std::shared_ptr<iir::Stmt>>& getChildren() { return ast_.getStatements(); }

  auto childrenBegin() { return ast_.getStatements().begin(); }
  auto childrenEnd() { return ast_.getStatements().end(); }

  inline auto childrenRBegin() { return ast_.getStatements().rbegin(); }
  inline auto childrenREnd() { return ast_.getStatements().rend(); }

  inline auto childrenBegin() const { return ast_.getStatements().begin(); }
  inline auto childrenEnd() const { return ast_.getStatements().end(); }

  inline auto childrenRBegin() const { return ast_.getStatements().rbegin(); }
  inline auto childrenREnd() const { return ast_.getStatements().rend(); }

  inline auto& getChild(unsigned long pos) { return ast_.getStatements()[pos]; }

  template <typename T>
  inline auto childrenErase(T childIt) {
    auto it_ = ast_.getStatements().erase(childIt);
    return it_;
  }

  inline bool checkTreeConsistency() const { return true; }

  /// @brief set the parent pointer of the children
  template <typename TChildSmartPtr>
  void setChildrenParent(TChildSmartPtr* = 0) {}

  void setChildParent(const std::shared_ptr<iir::Stmt>& child) {}

  void insertChild(std::shared_ptr<iir::Stmt>&& child) { ast_.getStatements().push_back(child); }

  void insertChild(std::shared_ptr<iir::Stmt>&& child, const std::unique_ptr<DoMethod>& thisNode) {
    ast_.getStatements().push_back(child);
  }

  auto insertChild(std::vector<std::shared_ptr<iir::Stmt>>::const_iterator pos,
                   std::shared_ptr<iir::Stmt>&& child) {
    return ast_.getStatements().insert(pos, std::move(child));
  }

  template <typename Iterator>
  auto insertChildren(std::vector<std::shared_ptr<iir::Stmt>>::const_iterator pos, Iterator first,
                      Iterator last) {
    return ast_.getStatements().insert(pos, first, last);
  }

  template <typename Iterator>
  ChildIterator insertChildren(ChildIterator pos, Iterator first, Iterator last,
                               const std::unique_ptr<DoMethod>&) {
    return ast_.getStatements().insert(pos, first, last);
  }

  void printTree() {}

  void replace(const std::shared_ptr<iir::Stmt>& inputChild,
               std::shared_ptr<iir::Stmt>& withNewChild) {
    auto it = std::find(ast_.getStatements().begin(), ast_.getStatements().end(), inputChild);
    *it = withNewChild;
  }

  void replace(const std::shared_ptr<iir::Stmt>& inputChild,
               std::shared_ptr<iir::Stmt>& withNewChild, const std::unique_ptr<DoMethod>&) {
    auto it = std::find(ast_.getStatements().begin(), ast_.getStatements().end(), inputChild);
    *it = withNewChild;
  }

  bool childrenEmpty() const { return ast_.getStatements().empty(); }
  void clearChildren() { ast_.getStatements().clear(); }

  void setAST(iir::BlockStmt&& ast) { ast_ = std::move(ast); }
};

} // namespace iir

template <typename RootNode>
auto iterateIIROverStmt(const RootNode& root) {
  std::vector<std::shared_ptr<iir::Stmt>> allStmts;
  for(auto& doMethod : iterateIIROver<iir::DoMethod>(root)) {
    std::copy(doMethod->getChildren().begin(), doMethod->getChildren().end(),
              std::back_inserter(allStmts));
  }
  return allStmts;
}
} // namespace dawn

#endif
