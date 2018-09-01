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

#ifndef DAWN_IIR_IIRNODE_H
#define DAWN_IIR_IIRNODE_H

#include "dawn/Support/Assert.h"
#include "dawn/Support/Unreachable.h"
#include "dawn/IIR/NodeUpdateType.h"
#include <vector>
#include <memory>
#include <type_traits>
#include <algorithm>
#include <iostream>
#include <list>
#include <iterator>

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

template <typename Parent, typename NodeType, typename Child,
          template <class> class Container = impl::StdVector>
class IIRNode {

protected:
  virtual ~IIRNode() = default;
  IIRNode() = default;
  IIRNode(IIRNode&&) = default;

  std::unique_ptr<Parent> const* parent_ = nullptr;

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

  inline void cloneChildrenFrom(const IIRNode& other) { cloneChildrenImpl<Child>(other); }

  inline void cloneChildrenFrom(const IIRNode& other, const SmartPtr<NodeType>& thisNode) {
    DAWN_ASSERT(thisNode.get() == this);
    cloneChildrenImpl<Child>(other, thisNode);
  }

  inline const Container<SmartPtr<Child>>& getChildren() const { return children_; }

  inline ChildIterator childrenBegin() { return children_.begin(); }
  inline ChildIterator childrenEnd() { return children_.end(); }

  inline typename Container<SmartPtr<Child>>::reverse_iterator childrenRBegin() {
    return children_.rbegin();
  }
  inline typename Container<SmartPtr<Child>>::reverse_iterator childrenREnd() {
    return children_.rend();
  }

  inline ChildConstIterator childrenBegin() const { return children_.begin(); }
  inline ChildConstIterator childrenEnd() const { return children_.end(); }

  inline typename Container<SmartPtr<Child>>::const_reverse_iterator childrenRBegin() const {
    return children_.rbegin();
  }
  inline typename Container<SmartPtr<Child>>::const_reverse_iterator childrenREnd() const {
    return children_.rend();
  }

  inline ChildIterator childrenErase(ChildIterator it) {
    auto it_ = children_.erase(it);

    if(!children_.empty()) {
      setChildParent<Parent, Child>(children_.back());
      updateFromChildrenRec<NodeType>();
    }

    return it_;
  }

  template <typename UnaryPredicate>
  inline bool childrenEraseIf(UnaryPredicate p) {
    bool res = RemoveIf(children_, p);

    if(!children_.empty()) {
      setChildParent<Parent, Child>(children_.back());
      updateFromChildrenRec<NodeType>();
    }
    return res;
  }

  inline const ChildSmartPtrType& getChild(unsigned long pos) {
    return getChildImpl<typename std::iterator_traits<ChildIterator>::iterator_category>(pos);
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
    return children_[it];
  }

  inline bool checkTreeConsistency() const { return checkTreeConsistencyImpl<Child>(); }

  virtual void updateFromChildren() {}

  inline const std::unique_ptr<Parent>& getParent() {
    DAWN_ASSERT(parent_);
    return *parent_;
  }

  inline const std::unique_ptr<Parent>* getParentPtr() { return parent_; }

  inline void setParent(const std::unique_ptr<Parent>& p) { parent_ = &p; }

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

  template <typename TChildSmartPtr, typename TNode>
  void
  setChildrenParent(const std::unique_ptr<TNode>& p,
                    typename std::enable_if<!is_child_void<TChildSmartPtr>::value>::type* = 0) {
    DAWN_ASSERT(parent_);
    const std::unique_ptr<NodeType>& thisNodeSmartPtr =
        (*parent_)->getChildSmartPtr(static_cast<NodeType*>(this));

    for(const auto& child : getChildren()) {
      child->setParent(thisNodeSmartPtr);
      child->template setChildrenParent<typename getChildType<TChildSmartPtr>::type::type>(child);
    }
  }

  template <typename TChildSmartPtr, typename TNode>
  void setChildrenParent(const std::unique_ptr<TNode>& p,
                         typename std::enable_if<is_child_void<TChildSmartPtr>::value>::type* = 0) {
  }

  template <typename TParent, typename TChild>
  void setChildParent(
      const std::unique_ptr<TChild>& child,
      typename std::enable_if<!std::is_void<TParent>::value &&
                              !is_child_void<std::unique_ptr<TChild>>::value>::type* = 0) {

    static_assert(std::is_same<TParent, Parent>::value,
                  "The template TParent type of this function should be == Parent. The function is "
                  "templated only for syntax specialization using SFINAE");

    if(parent_) {
      const std::unique_ptr<NodeType>& thisNodeSmartPtr =
          (*parent_)->getChildSmartPtr(static_cast<NodeType*>(this));

      child->setParent(thisNodeSmartPtr);

      // we recursively set the parent of not only this child but all the siblings, since an insert
      // into a vector can modify the pointers of all elements
      for(const auto& sibling : thisNodeSmartPtr->getChildren()) {
        sibling->setParent(thisNodeSmartPtr);
        sibling->template setChildrenParent<
            typename getChildType<std::unique_ptr<TChild>>::type::type>(child);
      }
    }
  }

  template <typename TParent, typename TChild>
  void setChildParent(
      const std::unique_ptr<TChild>& child,
      typename std::enable_if<std::is_void<TParent>::value ||
                              is_child_void<std::unique_ptr<TChild>>::value>::type* = 0) {}

  void insertChild(ChildSmartPtrType&& child) { insertChildImpl<Parent>(std::move(child)); }

  ChildIterator insertChild(ChildIterator pos, ChildSmartPtrType&& child) {
    return insertChildImpl<Parent>(pos, std::move(child));
  }

  template <typename Iterator>
  ChildIterator insertChildren(ChildIterator pos, Iterator first, Iterator last) {
    return insertChildrenImpl<Iterator, Parent>(pos, first, last);
  }

  template <typename Iterator, typename TChildParent>
  ChildIterator insertChildren(ChildIterator pos, Iterator first, Iterator last,
                               const std::unique_ptr<TChildParent>& p) {
    return insertChildrenImpl<Parent, Iterator, TChildParent>(pos, first, last, p);
  }

  void insertChild(ChildSmartPtrType&& child, const std::unique_ptr<NodeType>& p) {
    insertChildImpl<Parent, ChildSmartPtrType, std::unique_ptr<NodeType>>(std::move(child), p);
  }

  void printTree() { printTreeImpl<Child>(); }

  void replace(const SmartPtr<Child>& inputChild, SmartPtr<Child>& withNewChild) {
    replaceImpl<Parent>(inputChild, withNewChild);
  }

  void replace(const SmartPtr<Child>& inputChild, SmartPtr<Child>& withNewChild,
               const std::unique_ptr<NodeType>& thisNode) {
    replaceImpl<Parent>(inputChild, withNewChild, thisNode);
  }

  bool childrenEmpty() const { return children_.empty(); }

  void clearChildren() { children_.clear(); }

  inline const std::unique_ptr<Child>& getChildSmartPtr(Child* child) {

    for(const auto& c : children_) {
      if(c.get() == child) {
        return c;
      }
    }
    dawn_unreachable("child weak pointer not found");

    return *(children_.begin());
  }

  template <typename TNodeType>
  inline void updateFromChildrenRec(
      typename std::enable_if<std::is_void<typename TNodeType::ParentType>::value>::type* = 0) {

    static_cast<NodeType*>(this)->updateFromChildren();
  }

  template <typename TNodeType>
  inline void updateFromChildrenRec(
      typename std::enable_if<!std::is_void<typename TNodeType::ParentType>::value>::type* = 0) {

    static_cast<NodeType*>(this)->updateFromChildren();

    auto parentPtr = getParentPtr();
    if(parentPtr) {
      (*parentPtr)->template updateFromChildrenRec<typename TNodeType::ParentType>();
    }
  }

  void update(NodeUpdateType updateType) {
    if(impl::updateLevel(updateType)) {
      static_cast<NodeType*>(this)->updateLevel();
      if(!impl::updateTreeAbove(updateType)) {
        static_cast<NodeType*>(this)->updateFromChildren();
      }
    }
    if(impl::updateTreeAbove(updateType)) {
      updateFromChildrenRec<NodeType>();
    }
    if(impl::updateTreeBelow(updateType)) {
      dawn_unreachable("node update type tree below not supported");
    }
  }

  virtual void updateLevel() {}

private:
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

    bool result = true;
    for(const auto& child : getChildren()) {
      if(!child->parentIsSet()) {
        return false;
      }
      const auto& parent = child->getParent();

      if(parent.get() != this) {
        return false;
      }
      result = result & child->checkTreeConsistency();
    }

    return result;
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
      // TODO update
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

  template <typename TParent>
  void replaceImpl(const SmartPtr<Child>& inputChild, SmartPtr<Child>& withNewChild,
                   const std::unique_ptr<NodeType>& thisNode,
                   typename std::enable_if<std::is_void<TParent>::value>::type* = 0) {
    // TODO rethink this
    auto it = std::find(children_.begin(), children_.end(), inputChild);
    DAWN_ASSERT(it != children_.end());
    const std::unique_ptr<NodeType>* ptr = &((*it)->getParent());

    (it)->swap(withNewChild);

    inputChild->setParent(*ptr);
    inputChild
        ->template setChildrenParent<typename getChildType<std::unique_ptr<Child>>::type::type>(
            inputChild);

    updateFromChildrenRec<NodeType>();
  }

  template <typename TChild>
  void printTreeImpl(typename std::enable_if<!std::is_void<TChild>::value>::type* = 0) {
    for(const auto& child : getChildren()) {
      if(child->parentIsSet())
        std::cout << "&child : " << &child << "  child.get() : " << child.get()
                  << " child->getParentP() : " << parent_
                  << " parent.get() : " << child->getParent().get() << std::endl;
      else
        std::cout << "&child : " << &child << "  child.get() : " << child.get()
                  << " child->getParentP() : " << parent_ << " parent.get() : "
                  << "NULL" << std::endl;
    }
    for(const auto& child : children_) {
      child->printTree();
    }
  }

  template <typename TChild>
  void printTreeImpl(typename std::enable_if<std::is_void<TChild>::value>::type* = 0) {}
};

} // namespace iir
} // namespace dawn

#undef PROTECT_TEMPLATE

#endif
