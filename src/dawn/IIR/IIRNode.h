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

#include "dawn/Support/Unreachable.h"
#include <vector>
#include <memory>
#include <type_traits>

namespace dawn {
namespace iir {

template <typename Parent, typename NodeType, typename Child,
          template <class> class SmartPtr = std::shared_ptr>
class IIRNode {

protected:
  IIRNode() = default;
  IIRNode(const IIRNode&) = default;
  IIRNode(IIRNode&&) = default;

  std::weak_ptr<Parent> parent_;
  /// List of Do-Methods of this stage
  std::vector<SmartPtr<Child>> children_;

public:
  template <typename T>
  using child_smartptr_t = SmartPtr<T>;

  const std::vector<SmartPtr<Child>>& getChildren() const { return children_; }

  typename std::vector<SmartPtr<Child>>::iterator childrenBegin() { return children_.begin(); }
  typename std::vector<SmartPtr<Child>>::iterator childrenEnd() { return children_.end(); }

  typename std::vector<SmartPtr<Child>>::reverse_iterator childrenRBegin() {
    return children_.rbegin();
  }
  typename std::vector<SmartPtr<Child>>::reverse_iterator childrenREnd() {
    return children_.rend();
  }

  typename std::vector<SmartPtr<Child>>::const_iterator childrenBegin() const {
    return children_.begin();
  }
  typename std::vector<SmartPtr<Child>>::const_iterator childrenEnd() const {
    return children_.end();
  }

  typename std::vector<SmartPtr<Child>>::const_reverse_iterator childrenRBegin() const {
    return children_.rbegin();
  }
  typename std::vector<SmartPtr<Child>>::const_reverse_iterator childrenREnd() const {
    return children_.rend();
  }

  typename std::vector<SmartPtr<Child>>::iterator
  childrenErase(typename std::vector<SmartPtr<Child>>::iterator it) {
    return children_.erase(it);
  }

  std::weak_ptr<Child> getChildWeakPtr(Child* child) {
    for(const auto& c : children_) {
      if(c.get() == child)
        return c;
    }
    dawn_unreachable("child weak pointer not found");
  }

  template <typename T>
  void setChildParent(const SmartPtr<Child>& child,
                      typename std::enable_if<!std::is_void<T>::value>::type* = 0) {
    if(!parent_.expired()) {
      auto sparent = parent_.lock();
      std::weak_ptr<NodeType> p = sparent->getChildWeakPtr(this);
      child->setParent(p);
    }
  }

  template <typename T>
  void setChildParent(const SmartPtr<Child>& child,
                      typename std::enable_if<std::is_void<T>::value>::type* = 0) {}

  void insertChild(SmartPtr<Child>&& child) {
    setChildParent<Parent>(child);
    children_.push_back(std::move(child));
  }

  bool childrenEmpty() const { return children_.empty(); }

  void clearChildren() { children_.clear(); }
};

} // namespace iir
} // namespace dawn

#endif
