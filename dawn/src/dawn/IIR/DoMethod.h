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

  // TODO(SAP) removing getChildren() must be the last thing to do
  const std::vector<std::shared_ptr<iir::Stmt>>& getChildren() const {
    return ast_.getStatements();
  }
  std::vector<std::shared_ptr<iir::Stmt>>& getChildren() { return ast_.getStatements(); }
  // END_TODO

  // TODO(SAP) remove
  auto childrenBegin() {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements().begin();
  }
  // TODO(SAP) remove
  auto childrenEnd() {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements().end();
  }
  // TODO(SAP) remove
  inline auto childrenRBegin() {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements().rbegin();
  }
  // TODO(SAP) remove
  inline auto childrenREnd() {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements().rend();
  }
  // TODO(SAP) remove
  inline auto childrenBegin() const {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements().begin();
  }
  inline auto childrenEnd() const {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements().end();
  }
  // TODO(SAP) remove
  inline auto childrenRBegin() const {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements().rbegin();
  }
  // TODO(SAP) remove
  inline auto childrenREnd() const {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements().rend();
  }
  // TODO(SAP) remove
  inline auto& getChild(unsigned long pos) {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements()[pos];
  }
  // TODO(SAP) remove
  template <typename T>
  inline auto childrenErase(T childIt) {
    DAWN_ASSERT_MSG(false, "unreachable");
    auto it_ = ast_.getStatements().erase(childIt);
    return it_;
  }
  // TODO(SAP) remove
  inline bool checkTreeConsistency() const { return true; }

  // TODO(SAP) remove
  template <typename TChildSmartPtr>
  void setChildrenParent(TChildSmartPtr* = 0) {}
  // TODO(SAP) remove
  void setChildParent(const std::shared_ptr<iir::Stmt>& child) {}
  // TODO(SAP) remove
  void insertChild(std::shared_ptr<iir::Stmt>&& child) {
    DAWN_ASSERT_MSG(false, "unreachable");
    ast_.getStatements().push_back(child);
  }
  // TODO(SAP) remove
  void insertChild(std::shared_ptr<iir::Stmt>&& child, const std::unique_ptr<DoMethod>& thisNode) {
    DAWN_ASSERT_MSG(false, "unreachable");
    ast_.getStatements().push_back(child);
  }
  // TODO(SAP) remove
  auto insertChild(std::vector<std::shared_ptr<iir::Stmt>>::const_iterator pos,
                   std::shared_ptr<iir::Stmt>&& child) {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements().insert(pos, std::move(child));
  }
  // TODO(SAP) remove
  template <typename Iterator>
  auto insertChildren(std::vector<std::shared_ptr<iir::Stmt>>::const_iterator pos, Iterator first,
                      Iterator last) {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements().insert(pos, first, last);
  }
  // TODO(SAP) remove
  template <typename Iterator>
  ChildIterator insertChildren(ChildIterator pos, Iterator first, Iterator last,
                               const std::unique_ptr<DoMethod>&) {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements().insert(pos, first, last);
  }

  void printTree() {}
  // TODO(SAP) remove
  void replace(const std::shared_ptr<iir::Stmt>& inputChild,
               std::shared_ptr<iir::Stmt>& withNewChild) {
    DAWN_ASSERT_MSG(false, "unreachable");
    auto it = std::find(ast_.getStatements().begin(), ast_.getStatements().end(), inputChild);
    *it = withNewChild;
  }
  // TODO(SAP) remove
  void replace(const std::shared_ptr<iir::Stmt>& inputChild,
               std::shared_ptr<iir::Stmt>& withNewChild, const std::unique_ptr<DoMethod>&) {
    DAWN_ASSERT_MSG(false, "unreachable");
    auto it = std::find(ast_.getStatements().begin(), ast_.getStatements().end(), inputChild);
    *it = withNewChild;
  }

  // TODO(SAP) remove
  bool childrenEmpty() const {
    DAWN_ASSERT_MSG(false, "unreachable");
    return ast_.getStatements().empty();
  }
  // TODO(SAP) remove
  void clearChildren() {
    DAWN_ASSERT_MSG(false, "unreachable");
    ast_.getStatements().clear();
  }

  void setAST(iir::BlockStmt&& ast) { ast_ = std::move(ast); }
  iir::BlockStmt const& getAST() const { return ast_; }
  iir::BlockStmt& getAST() { return ast_; }
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
