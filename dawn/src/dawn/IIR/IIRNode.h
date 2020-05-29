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

#include "dawn/IIR/NodeUpdateType.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/RemoveIf.hpp"
#include "dawn/Support/Unreachable.h"
#include <algorithm>
#include <iterator>
#include <list>
#include <memory>
#include <ostream>
#include <type_traits>
#include <vector>

#ifndef PROTECT_TEMPLATE
#define PROTECT_TEMPLATE(TEMP, TYPE)                                                               \
  static_assert(std::is_same<TEMP, TYPE>::value,                                                   \
                "The template TEMP type of this function should be == TYPE. The function is "      \
                "templated only for syntax specialization using SFINAE");
#endif

namespace dawn {
namespace iir {

namespace impl {
template <typename T>
using StdVector = std::vector<T, std::allocator<T>>;
}

template <typename P>
bool ptrEqual(const std::weak_ptr<P>& f, const std::weak_ptr<P>& s) {
  return !f.expired() && f.lock() == s.lock();
}

/// @class base class of a node of the IIR tree
/// @tparam Parent parent class of the node
/// @tparam NodeType the same class that inherits from this IIRNode, i.e. this node
/// @tparam Child child class of the node
/// @tparam Container stl containter that stores the children
template <typename Parent, typename NodeType, typename Child,
          template <class> class Container = impl::StdVector>
class IIRNode {

protected:
  /// @brief constructors
  /// @{
  virtual ~IIRNode() = default;
  /// @}

  const std::unique_ptr<Parent>* parent_ = nullptr;

  template <class T>
  using SmartPtr = typename std::conditional<std::is_void<Child>::value, std::shared_ptr<T>,
                                             std::unique_ptr<T>>::type;

  Container<SmartPtr<Child>> children_;

public:
  using ParentType = Parent;
  using ChildType = Child;
  using ChildSmartPtrType = SmartPtr<Child>;
  using ChildrenContainer = Container<SmartPtr<Child>>;
  using ChildIterator = typename Container<SmartPtr<Child>>::iterator;
  using ChildConstIterator = typename Container<SmartPtr<Child>>::const_iterator;

  template <typename T>
  using child_smartptr_t = SmartPtr<T>;
  using child_iterator_t = typename Container<SmartPtr<Child>>::iterator;
  using child_reverse_iterator_t = typename Container<SmartPtr<Child>>::reverse_iterator;

  /// @brief clone the children of another node and store them as children of this object
  /// @param other node from where the children are cloned
  inline void cloneChildrenFrom(const IIRNode& other) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    cloneChildrenImpl<Child>(other);
  }
  /// @brief clone the children of another node and store them as children of this object
  /// @param other node from where the children are cloned
  /// @param thisNode smart ptr of this object (specialization for nodes that have no parent)
  inline void cloneChildrenFrom(const IIRNode& other, const SmartPtr<NodeType>& thisNode) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    DAWN_ASSERT(thisNode.get() == this);
    cloneChildrenImpl<Child>(other, thisNode);
  }

  /// @brief getters and iterator getters
  /// @{
  inline const Container<SmartPtr<Child>>& getChildren() const {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return children_;
  }
  inline Container<SmartPtr<Child>>& getChildren() {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return children_;
  }

  inline ChildIterator childrenBegin() {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return children_.begin();
  }
  inline ChildIterator childrenEnd() {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return children_.end();
  }

  /// @brief reverse iterator begin
  inline typename Container<SmartPtr<Child>>::reverse_iterator childrenRBegin() {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return children_.rbegin();
  }
  /// @brief reverse iterator end
  inline typename Container<SmartPtr<Child>>::reverse_iterator childrenREnd() {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return children_.rend();
  }

  /// @brief iterator begin
  inline ChildConstIterator childrenBegin() const {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return children_.begin();
  }
  /// @brief iterator end
  inline ChildConstIterator childrenEnd() const {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return children_.end();
  }

  inline typename Container<SmartPtr<Child>>::const_reverse_iterator childrenRBegin() const {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return children_.rbegin();
  }
  inline typename Container<SmartPtr<Child>>::const_reverse_iterator childrenREnd() const {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return children_.rend();
  }
  inline const ChildSmartPtrType& getChild(unsigned long pos) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return getChildImpl<typename std::iterator_traits<ChildIterator>::iterator_category>(pos);
  }
  /// @brief get unique_ptr to parent
  inline const std::unique_ptr<Parent>& getParent() const {
    DAWN_ASSERT(parent_);
    return *parent_;
  }

  /// @brief get raw pointer to parent smart ptr
  inline const std::unique_ptr<Parent>* getParentPtr() { return parent_; }
  /// @}

  /// @brief erase a children element (tree consistency is ensured after erasure)
  /// @param childIt iterator pointing to the element being erased
  /// @return iterator to next element
  inline ChildIterator childrenErase(ChildIterator childIt) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    auto it_ = children_.erase(childIt);

    fixAfterErase();

    return it_;
  }

  /// @brief conditional erase a children element if the predicate returns true
  /// (tree consistency is ensured after erasure)
  /// @param pred predicate used to find the element that will be deleted
  /// @return true if element is found and deleted
  template <typename UnaryPredicate>
  inline bool childrenEraseIf(UnaryPredicate pred) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    bool res = RemoveIf(children_, pred);

    if(!res)
      return res;

    fixAfterErase();
    return res;
  }

  /// @brief check the consistency of the tree
  /// Every tree node contains pointers to parent node. When the tree is modified, i.e. a node is
  /// inserted or deleted, the pointers pointing to parent nodes can be invalidated. This method
  /// checks if all these pointers are valid
  inline bool checkTreeConsistency() const { return checkTreeConsistencyImpl<Child>(); }

  /// @brief virtual method to be implemented by node classes that update the derived info from
  /// children derived infos
  virtual void updateFromChildren() {}

  inline void setParent(const std::unique_ptr<Parent>& p) { parent_ = &p; }

  /// @brief check if the pointer to parent is set
  inline bool parentIsSet() const { return static_cast<bool>(parent_); }

  template <bool T>
  struct identity {
    using type = std::integral_constant<bool, T>;
  };

  template <typename T>
  struct identity_t {
    using type = T;
  };

  template <typename T>
  struct isVoid {
    using type = typename std::is_void<typename std::remove_reference<T>::type::element_type>::type;
  };

  template <typename T>
  struct is_child_void {
    static constexpr bool value =
        std::conditional<std::is_void<T>::value, identity<true>, isVoid<T>>::type::type::value;
  };

  template <typename T>
  struct getChildTypeImpl {
    using type = typename std::remove_reference<T>::type::element_type::ChildSmartPtrType;
  };

  template <typename T>
  struct getChildType {
    using type = typename std::conditional<std::is_void<T>::value, identity_t<void>,
                                           getChildTypeImpl<T>>::type;
  };

  /// @brief set the parent pointer of the children
  template <typename TChildSmartPtr>
  void
  setChildrenParent(typename std::enable_if<!is_child_void<TChildSmartPtr>::value>::type* = 0) {
    if(!parent_)
      return;

    const std::unique_ptr<NodeType>& thisNodeSmartPtr =
        (*parent_)->getChildSmartPtr(static_cast<NodeType*>(this));

    for(const auto& child : getChildren()) {
      child->setParent(thisNodeSmartPtr);
      child->template setChildrenParent<typename getChildType<TChildSmartPtr>::type::type>();
    }
  }

  /// @brief set the parent pointer of the children
  template <typename TChildSmartPtr>
  void setChildrenParent(typename std::enable_if<is_child_void<TChildSmartPtr>::value>::type* = 0) {
  }

  /// @brief set the parent pointer of a child of this node
  /// @param child child node of this node, for which the parent pointer will be set
  template <typename TParent, typename TChild>
  void setChildParent(
      const std::unique_ptr<TChild>& child,
      typename std::enable_if<!std::is_void<TParent>::value &&
                              !is_child_void<std::unique_ptr<TChild>>::value>::type* = 0) {

    static_assert(std::is_same<TParent, Parent>::value,
                  "The template TParent type of this function should be == Parent. The function is "
                  "templated only for syntax specialization using SFINAE");

    if(parent_) {
      // we need to find the smart ptr of "this"
      const std::unique_ptr<NodeType>& thisNodeSmartPtr =
          (*parent_)->getChildSmartPtr(static_cast<NodeType*>(this));

      child->setParent(thisNodeSmartPtr);

      // we recursively set the parent of not only this child but all the siblings, since an insert
      // into a vector can modify the pointers of all elements
      for(const auto& sibling : thisNodeSmartPtr->getChildren()) {
        sibling->setParent(thisNodeSmartPtr);
        sibling->template setChildrenParent<
            typename getChildType<std::unique_ptr<TChild>>::type::type>();
      }
    }
  }

  template <typename TParent, typename TChild>
  void setChildParent(
      const std::unique_ptr<TChild>& child,
      typename std::enable_if<std::is_void<TParent>::value ||
                              is_child_void<std::unique_ptr<TChild>>::value>::type* = 0) {}

  /// @brief insert a child node (specialization for nodes with a parent node)
  /// @param child node being inserted as a child
  void insertChild(ChildSmartPtrType&& child) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    insertChildImpl<Parent>(std::move(child));
  }

  /// @brief insert a child node (specialization for nodes without a parent node)
  /// @param child node being inserted as a child
  /// @param thisNode smart ptr to this
  void insertChild(ChildSmartPtrType&& child, const std::unique_ptr<NodeType>& thisNode) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    insertChildImpl<Parent, ChildSmartPtrType, std::unique_ptr<NodeType>>(std::move(child),
                                                                          thisNode);
  }

  /// @brief insert a child node
  /// @param pos iterator with the position where the child will be inserted
  /// @param child node being inserted as a child
  ChildIterator insertChild(ChildIterator pos, ChildSmartPtrType&& child) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return insertChildImpl<Parent>(pos, std::move(child));
  }

  /// @brief insert children within an iterator range (specialization for nodes with a parent node)
  /// @param pos iterator with the position where the child will be inserted
  /// @param first iterator to begin of the children being inserted
  /// @param last iterator to end of the children being inserted
  template <typename Iterator>
  ChildIterator insertChildren(ChildIterator pos, Iterator first, Iterator last) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return insertChildrenImpl<Iterator, Parent>(pos, first, last);
  }

  /// @brief insert children within an iterator range (specialization for nodes without a parent
  /// node)
  /// @param pos iterator with the position where the child will be inserted
  /// @param first iterator to begin of the children being inserted
  /// @param last iterator to end of the children being inserted
  template <typename Iterator, typename TChildParent>
  ChildIterator insertChildren(ChildIterator pos, Iterator first, Iterator last,
                               const std::unique_ptr<TChildParent>& p) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return insertChildrenImpl<Parent, Iterator, TChildParent>(pos, first, last, p);
  }

  /// @brief print tree of pointers
  void printTree(std::ostream& os) { printTreeImpl<Child>(os); }

  /// @brief replace a child node by another node (specialization for nodes with a parent node)
  /// @param inputChild child node that will be looked up and replaced
  /// @param withNewChild new child node that will be inserted in the place of the old node
  void replace(const SmartPtr<Child>& inputChild, SmartPtr<Child>& withNewChild) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    replaceImpl<Parent>(inputChild, withNewChild);
  }

  /// @brief replace a child node by another node (specialization for nodes without a parent node)
  /// @param inputChild child node that will be looked up and replaced
  /// @param withNewChild new child node that will be inserted in the place of the old node
  /// @param thisNode smart ptr to this node
  void replace(const SmartPtr<Child>& inputChild, SmartPtr<Child>& withNewChild,
               const std::unique_ptr<NodeType>& thisNode) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    replaceImpl<Parent>(inputChild, withNewChild, thisNode);
  }

  /// @brief true if there are no children
  bool childrenEmpty() const {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    return children_.empty();
  }

  /// @brief clear the container of children
  void clearChildren() {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    children_.clear();
  }

  /// @brief get the smart pointer of a raw pointer child node
  inline const std::unique_ptr<Child>& getChildSmartPtr(Child* child) {
    static_assert(!std::is_void<Child>::value, "Child type must not be void");
    for(const auto& c : children_) {
      if(c.get() == child) {
        return c;
      }
    }
    dawn_unreachable("child weak pointer not found");

    return *(children_.begin());
  }

  /// @brief update recursively (propagating to the top of the tree) the derived info of this node
  template <typename TNodeType>
  inline void updateFromChildrenRec(
      typename std::enable_if<std::is_void<typename TNodeType::ParentType>::value>::type* = 0) {
    updateFromChildren();
  }

  /// @brief update recursively (propagating to the top of the tree) the derived info of this node
  template <typename TNodeType>
  inline void updateFromChildrenRec(
      typename std::enable_if<!std::is_void<typename TNodeType::ParentType>::value>::type* = 0) {
    updateFromChildren();

    auto parentPtr = getParentPtr();
    if(parentPtr) {
      (*parentPtr)->template updateFromChildrenRec<typename TNodeType::ParentType>();
    }
  }

  /// @brief clear the derived info recursively (propagating to the top of the tree)
  template <typename TNodeType>
  inline void clearDerivedInfoRec(
      typename std::enable_if<std::is_void<typename TNodeType::ParentType>::value>::type* = 0) {
    clearDerivedInfo();
  }

  /// @brief clear the derived info recursively (propagating to the top of the tree)
  template <typename TNodeType>
  inline void clearDerivedInfoRec(
      typename std::enable_if<!std::is_void<typename TNodeType::ParentType>::value>::type* = 0) {

    auto parentPtr = getParentPtr();
    if(parentPtr) {
      (*parentPtr)->clearDerivedInfo();
      (*parentPtr)->template clearDerivedInfoRec<typename TNodeType::ParentType>();
    }
  }

  /// @brief update the derived info of the node
  /// @param updateType determines if the update should be applied to this tree level (only) or
  /// propagate it to the top or bottom of the tree
  void update(NodeUpdateType updateType) {
    if(impl::updateLevel(updateType)) {
      clearDerivedInfo();
      static_cast<NodeType*>(this)->updateLevel();
      if(!impl::updateTreeAbove(updateType)) {
        updateFromChildren();
      }
    }
    if(impl::updateTreeAbove(updateType)) {
      clearDerivedInfoRec<NodeType>();
      updateFromChildrenRec<NodeType>();
    }
    if(impl::updateTreeBelow(updateType)) {
      dawn_unreachable("node update type tree below not supported");
    }
  }

  virtual void updateLevel() {}
  virtual void clearDerivedInfo() {}

private:
  /// @brief fix the tree structure after the erase of a child
  inline void fixAfterErase() {
    // since we have removed a child, the pointers of other siblings might have change,
    // there we need to reset the parent pointer of grandchildren
    for(auto& childIt_ : children_) {
      childIt_->template setChildrenParent<typename Child::ChildSmartPtrType>();
    }
    // recompute the derived info from the remaining children
    if(!children_.empty()) {
      updateFromChildrenRec<NodeType>();
    }
  }

  template <typename IteratorCategory>
  inline const ChildSmartPtrType& getChildImpl(
      unsigned long pos,
      typename std::enable_if<
          std::is_same<IteratorCategory, std::random_access_iterator_tag>::value>::type* = 0) {
    return children_[pos];
  }

  template <typename IteratorCategory>
  inline const ChildSmartPtrType& getChildImpl(
      unsigned long pos,
      typename std::enable_if<
          !std::is_same<IteratorCategory, std::random_access_iterator_tag>::value>::type* = 0) {

    auto it = std::next(children_.begin(), pos);
    return *it;
  }

  template <typename TChild>
  void cloneChildrenImpl(const IIRNode& other,
                         typename std::enable_if<std::is_void<TChild>::value>::type* = 0) {}

  template <typename TChild>
  void cloneChildrenImpl(const IIRNode& other,
                         typename std::enable_if<!std::is_void<TChild>::value>::type* = 0) {
    static_assert(std::is_same<TChild, Child>::value,
                  "The template TParent type of this function should be == Parent. The function is "
                  "templated only for syntax specialization using SFINAE");

    for(const auto& child : other.getChildren()) {
      insertChild(child->clone());
    }
  }

  template <typename TChild>
  void cloneChildrenImpl(const IIRNode& other, const SmartPtr<NodeType>& thisNode,
                         typename std::enable_if<std::is_void<TChild>::value>::type* = 0) {}

  template <typename TChild>
  void cloneChildrenImpl(const IIRNode& other, const SmartPtr<NodeType>& thisNode,
                         typename std::enable_if<!std::is_void<TChild>::value>::type* = 0) {
    for(const auto& child : other.getChildren()) {
      insertChild(child->clone(), thisNode);
    }
  }

  template <typename TChild>
  bool
  checkTreeConsistencyImpl(typename std::enable_if<std::is_void<TChild>::value>::type* = 0) const {
    return true;
  }

  template <typename TChild>
  bool
  checkTreeConsistencyImpl(typename std::enable_if<!std::is_void<TChild>::value>::type* = 0) const {
    PROTECT_TEMPLATE(TChild, Child)

    for(const auto& child : getChildren()) {
      if(!child->parentIsSet()) {
        return false;
      }
      const auto& parent = child->getParent();

      if(parent.get() != this) {
        return false;
      }
      return child->checkTreeConsistency();
    }

    return true;
  }

  template <typename TParent>
  void insertChildImpl(ChildSmartPtrType&& child,
                       typename std::enable_if<!std::is_void<TParent>::value>::type* = 0) {
    PROTECT_TEMPLATE(TParent, Parent)
    children_.push_back(std::move(child));
    repairTreeOfChildren();
  }

  template <typename TParent>
  ChildIterator insertChildImpl(ChildIterator pos, ChildSmartPtrType&& child,
                                typename std::enable_if<!std::is_void<TParent>::value>::type* = 0) {

    PROTECT_TEMPLATE(TParent, Parent)
    auto it = children_.insert(pos, std::move(child));

    repairTreeOfChildren();
    return it;
  }

  template <typename Iterator, typename TParent>
  ChildIterator
  insertChildrenImpl(ChildIterator pos, Iterator first, Iterator last,
                     typename std::enable_if<!std::is_void<TParent>::value>::type* = 0) {
    PROTECT_TEMPLATE(TParent, Parent)

    auto newfirst = children_.insert(pos, first, last);

    repairTreeOfChildren();
    return newfirst;
  }

  void repairTreeOfChildren(const std::unique_ptr<NodeType>& p) {
    for(const auto& sibling : children_) {
      sibling->setParent(p);

      if(!sibling->getChildren().empty()) {
        const auto& lastSibling = sibling->getChildren().back();
        sibling->template setChildParent<NodeType>(lastSibling);
      }
    }

    updateFromChildrenRec<NodeType>();
  }

  void repairTreeOfChildren() {
    // setting recursively the children parents is performed on all the siblings of a child,
    // therefore
    // here we only need to do it on one child
    const auto& lastChild = children_.back();
    setChildParent<Parent>(lastChild);

    updateFromChildrenRec<NodeType>();
  }

  template <typename TParent, typename Iterator, typename TChildParent>
  ChildIterator
  insertChildrenImpl(ChildIterator pos, Iterator first, Iterator last,
                     const std::unique_ptr<NodeType>& p,
                     typename std::enable_if<std::is_void<TParent>::value &&
                                             !std::is_void<TChildParent>::value>::type* = 0) {
    PROTECT_TEMPLATE(TParent, Parent)
    PROTECT_TEMPLATE(TChildParent, NodeType)

    auto newfirst = children_.insert(pos, first, last);

    repairTreeOfChildren(p);

    return newfirst;
  }

  template <typename TParent, typename TChild, typename TNodeSmartPtr>
  void insertChildImpl(TChild&& child, const TNodeSmartPtr& p,
                       typename std::enable_if<std::is_void<TParent>::value &&
                                               !std::is_void<TChild>::value>::type* = 0) {

    DAWN_ASSERT(p.get() == this);
    PROTECT_TEMPLATE(TParent, Parent)
    children_.push_back(std::move(child));

    repairTreeOfChildren(p);
  }

  template <typename TParent>
  void replaceImpl(const SmartPtr<Child>& inputChild, SmartPtr<Child>& withNewChild,
                   typename std::enable_if<!std::is_void<TParent>::value>::type* = 0) {
    auto it = std::find(children_.begin(), children_.end(), inputChild);
    DAWN_ASSERT(it != children_.end());
    const std::unique_ptr<NodeType>* ptr = &((*it)->getParent());

    (it)->swap(withNewChild);
    inputChild->setParent(*ptr);
    setChildParent<Parent, Child>(inputChild);

    updateFromChildrenRec<NodeType>();
  }

  /// @brief replace a child node by another node (specialization for nodes that do not have a
  /// parent)
  /// @param inputChild child node that will be looked up and replaced
  /// @param withNewChild new child node that will be inserted in the place of the old node
  /// @param thisNode smart ptr to this node
  template <typename TParent>
  void replaceImpl(const SmartPtr<Child>& inputChild, SmartPtr<Child>& withNewChild,
                   const std::unique_ptr<NodeType>& thisNode,
                   typename std::enable_if<std::is_void<TParent>::value>::type* = 0) {
    auto it = std::find(children_.begin(), children_.end(), inputChild);
    DAWN_ASSERT(it != children_.end());
    const std::unique_ptr<NodeType>* ptr = &((*it)->getParent());

    (it)->swap(withNewChild);

    inputChild->setParent(*ptr);
    inputChild
        ->template setChildrenParent<typename getChildType<std::unique_ptr<Child>>::type::type>();

    updateFromChildrenRec<NodeType>();
  }

  /// @brief print the tree of pointers (for debugging)
  template <typename TChild>
  void printTreeImpl(std::ostream& os,
                     typename std::enable_if<!std::is_void<TChild>::value>::type* = 0) {
    for(const auto& child : getChildren()) {
      if(child->parentIsSet())
        os << "&child : " << &child << "  child.get() : " << child.get()
           << " child->getParentP() : " << parent_ << " parent.get() : " << child->getParent().get()
           << "\n";
      else
        os << "&child : " << &child << "  child.get() : " << child.get()
           << " child->getParentP() : " << parent_ << " parent.get() : "
           << "NULL"
           << "\n";
    }
    for(const auto& child : children_) {
      child->printTree(os);
    }
  }

  template <typename TChild>
  void printTreeImpl(std::ostream& os,
                     typename std::enable_if<std::is_void<TChild>::value>::type* = 0) {}
};

} // namespace iir
} // namespace dawn

#undef PROTECT_TEMPLATE
