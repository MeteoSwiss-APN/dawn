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

#include <vector>
#include <memory>

namespace dawn {
namespace iir {

template <typename Parent, typename Child>
class IIRNode {

protected:
  /// List of Do-Methods of this stage
  std::vector<std::unique_ptr<Child>> children_;

public:
  const std::vector<std::unique_ptr<Child>>& getChildren() const { return children_; }

  typename std::vector<std::unique_ptr<Child>>::iterator childrenBegin() {
    return children_.begin();
  }
  typename std::vector<std::unique_ptr<Child>>::iterator childrenEnd() { return children_.end(); }

  typename std::vector<std::unique_ptr<Child>>::reverse_iterator childrenRBegin() {
    return children_.rbegin();
  }
  typename std::vector<std::unique_ptr<Child>>::reverse_iterator childrenREnd() {
    return children_.rend();
  }

  typename std::vector<std::unique_ptr<Child>>::const_iterator childrenBegin() const {
    return children_.begin();
  }
  typename std::vector<std::unique_ptr<Child>>::const_iterator childrenEnd() const {
    return children_.end();
  }

  typename std::vector<std::unique_ptr<Child>>::const_reverse_iterator childrenRBegin() const {
    return children_.rbegin();
  }
  typename std::vector<std::unique_ptr<Child>>::const_reverse_iterator childrenREnd() const {
    return children_.rend();
  }

  typename std::vector<std::unique_ptr<Child>>::iterator
  childrenErase(typename std::vector<std::unique_ptr<Child>>::iterator it) {
    return children_.erase(it);
  }

  void insertChild(std::unique_ptr<Child>&& child) { children_.push_back(std::move(child)); }

  bool childrenEmpty() const { return children_.empty(); }

  void clearChildren() { children_.clear(); }
};

} // namespace iir
} // namespace dawn

#endif
