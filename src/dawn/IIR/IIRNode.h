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
#include <vector>
#include <memory>
#include <type_traits>

#include <iostream>
#include <list>
#include <iterator>

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
          template <class> class SmartPtr = std::shared_ptr,
          template <class> class Container = impl::StdVector>
class IIRNode {

protected:
  IIRNode() = default;
  // TODO remove copy ctr, should not be used
  IIRNode(const IIRNode&) = default;
  IIRNode(IIRNode&&) = default;

  std::weak_ptr<Parent> parent_;
  /// List of Do-Methods of this stage
  Container<SmartPtr<Child>> children_;

public:
  using ChildrenContainer = Container<SmartPtr<Child>>;
  using ChildIterator = typename Container<SmartPtr<Child>>::iterator;
  using ChildConstIterator = typename Container<SmartPtr<Child>>::const_iterator;

  template <typename T>
  using child_smartptr_t = SmartPtr<T>;
  using child_iterator_t = typename Container<SmartPtr<Child>>::iterator;
  using child_reverse_iterator_t = typename Container<SmartPtr<Child>>::reverse_iterator;

  template <typename TChild>
  void cloneChildrenImpl(const IIRNode& other,
                         typename std::enable_if<std::is_void<TChild>::value>::type* = 0) {}

  template <typename TChild>
  void cloneChildrenImpl(const IIRNode& other,
                         typename std::enable_if<!std::is_void<TChild>::value>::type* = 0) {
    for(const auto& child : other.getChildren()) {
      insertChild(child->clone());
    }
  }

  void cloneChildren(const IIRNode& other) { cloneChildrenImpl<Child>(other); }

  Container<SmartPtr<Child>>& getChildren() { return children_; }
  const Container<SmartPtr<Child>>& getChildren() const { return children_; }

  // TODO alias for iterators
  //  typename Container<SmartPtr<Child>>::iterator childrenBegin() { return children_.begin(); }
  //  typename Container<SmartPtr<Child>>::iterator childrenEnd() { return children_.end(); }

  ChildIterator childrenBegin() { return children_.begin(); }
  ChildIterator childrenEnd() { return children_.end(); }

  typename Container<SmartPtr<Child>>::reverse_iterator childrenRBegin() {
    return children_.rbegin();
  }
  typename Container<SmartPtr<Child>>::reverse_iterator childrenREnd() { return children_.rend(); }

  ChildConstIterator childrenBegin() const { return children_.begin(); }
  ChildConstIterator childrenEnd() const { return children_.end(); }

  typename Container<SmartPtr<Child>>::const_reverse_iterator childrenRBegin() const {
    return children_.rbegin();
  }
  typename Container<SmartPtr<Child>>::const_reverse_iterator childrenREnd() const {
    return children_.rend();
  }

  ChildIterator childrenErase(ChildIterator it) { return ChildIterator{children_.erase(it)}; }

  std::weak_ptr<Child> getChildWeakPtr(Child* child) {

    // TODO move the parent to const unique_ptr
    //    for(const auto& c : children_) {
    //      if(c.get() == child)
    //        return c;
    //    }
    return std::weak_ptr<Child>();
    dawn_unreachable("child weak pointer not found");
  }

  template <typename TChild>
  void checkConsistencyImpl(typename std::enable_if<std::is_void<TChild>::value>::type* = 0) const {
  }

  template <typename TChild>
  void
  checkConsistencyImpl(typename std::enable_if<!std::is_void<TChild>::value>::type* = 0) const {
    for(const auto& child : getChildren()) {
      auto weak = child->getParent();
      DAWN_ASSERT(!weak.expired());

      child->checkConsistency();
    }
  }

  void checkConsistency() const { checkConsistencyImpl<Child>(); }

  std::weak_ptr<Parent> getParent() { return parent_; }

  void setParent(std::weak_ptr<Parent> p) { parent_ = p; }

  template <typename TChild, typename TParent>
  void setChildParent(TChild&& child,
                      typename std::enable_if<!std::is_void<TParent>::value>::type* = 0) {
    if(!parent_.expired()) {
      auto sparent = parent_.lock();
      std::weak_ptr<NodeType> p = sparent->getChildWeakPtr(static_cast<NodeType*>(this));
      child->setParent(p);
    }
  }

  template <typename TChild, typename TParent>
  void setChildParent(TChild&& child,
                      typename std::enable_if<std::is_void<TParent>::value>::type* = 0) {}

  template <typename TChild, typename TParent>
  void setChildParent(TChild const& child,
                      typename std::enable_if<!std::is_void<TParent>::value>::type* = 0) {
    if(!parent_.expired()) {
      auto sparent = parent_.lock();
      std::weak_ptr<NodeType> p = sparent->getChildWeakPtr(static_cast<NodeType*>(this));
      child->setParent(p);
    }
  }

  template <typename TChild, typename TParent>
  void setChildParent(TChild const& child,
                      typename std::enable_if<std::is_void<TParent>::value>::type* = 0) {}

  template <typename TChild, typename TParent>
  void insertChildImpl(TChild&& child,
                       typename std::enable_if<!std::is_void<TParent>::value>::type* = 0) {
    static_assert((std::is_same<TParent, Parent>::value), "not the same parent types");
    setChildParent<TChild, Parent>(child);
    children_.push_back(std::move(child));
  }

  template <typename TChild>
  void insertChild(TChild&& child) {
    insertChildImpl<TChild, Parent>(std::forward<TChild>(child));
  }

  template <typename TChild>
  ChildIterator insertChild(ChildIterator pos, TChild&& child) {
    return insertChildImpl<TChild, Parent>(pos, std::forward<TChild>(child));
  }

  template <typename TChild, typename TParent>
  ChildIterator insertChildImpl(ChildIterator pos, TChild&& child,
                                typename std::enable_if<!std::is_void<TParent>::value>::type* = 0) {
    // TODO watch out this first forward
    setChildParent<TChild, Parent>(std::forward<TChild>(child));
    return children_.insert(pos, std::forward<TChild>(child));
  }

  template <typename Iterator>
  ChildIterator insertChildren(ChildIterator pos, Iterator first, Iterator last) {
    return insertChildrenImpl<Iterator, Parent>(pos, first, last);
  }

  // TODO make specialization for move_iterator
  template <typename Iterator, typename TParent>
  ChildIterator
  insertChildrenImpl(ChildIterator pos, Iterator first, Iterator last,
                     typename std::enable_if<!std::is_void<TParent>::value>::type* = 0) {
    for(auto it = first; it != last; ++it) {
      setChildParent<SmartPtr<Child>, Parent>(*it);
    }
    return children_.insert(pos, first, last);
  }

  // TODO make Iterator first & last not templated
  template <typename Iterator, template <class> class TSmartPtrP>
  ChildIterator insertChildren(ChildIterator pos, Iterator first, Iterator last,
                               TSmartPtrP<NodeType> const& p) {
    for(auto it = first; it != last; ++it) {
      (*it)->setParent(p);
    }
    return children_.insert(pos, first, last);
  }

  template <typename TChild, template <class> class TSmartPtrP>
  void insertChild(TChild&& child, TSmartPtrP<NodeType> const& p) {
    child->setParent(p);
    children_.push_back(std::move(child));
  }

  bool childrenEmpty() const { return children_.empty(); }

  void clearChildren() { children_.clear(); }
};

} // namespace iir
} // namespace dawn

#endif
