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

#include "dawn/Support/Assert.h"
#include "dawn/Support/Unreachable.h"
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>

namespace dawn {
namespace iir {

template <typename RootIIRNode, typename LeafNode>
class IIRNodeIterator {
  template <class A, class B>
  friend class IIRNodeIterator;

  typename RootIIRNode::ChildConstIterator iterator_;
  typename RootIIRNode::ChildConstIterator end_;
  IIRNodeIterator<typename RootIIRNode::ChildType, LeafNode> restIterator_;

  // the iterator is a void iterator if the node contains no children
  bool voidIter_ = false;
  std::string name = RootIIRNode::name;
  const RootIIRNode* root_;
  bool isTop_ = false;

public:
  /// @brief IIRNodeIterator constructor
  /// @param root IIRNodeIterator will model iterators for children of this node
  /// @param isTop true if this is the top node of the multi-iterator call
  IIRNodeIterator(const RootIIRNode* root, bool isTop = false) : root_(root), isTop_(isTop) {
    DAWN_ASSERT(root);
    iterator_ = (root->childrenBegin());
    end_ = (root->childrenEnd());
    if(!root->childrenEmpty()) {
      restIterator_ =
          IIRNodeIterator<typename RootIIRNode::ChildType, LeafNode>(root->childrenBegin()->get());
    } else {
      voidIter_ = true;
    }
  }

  bool isVoidIter() const {
    if(voidIter_)
      return true;
    return restIterator_.isVoidIter();
  }

private:
  IIRNodeIterator() = default;

public:
  IIRNodeIterator(IIRNodeIterator&&) = default;
  IIRNodeIterator(const IIRNodeIterator&) = default;
  IIRNodeIterator& operator=(IIRNodeIterator&&) = default;

  IIRNodeIterator& operator++() {
    if(voidIter_)
      return *this;

    ++restIterator_;

    if(restIterator_.isEnd()) {
      ++iterator_;

      if(!isEnd()) {
        restIterator_ =
            IIRNodeIterator<typename RootIIRNode::ChildType, LeafNode>(iterator_->get());
      }
    }
    // If the restIterator contains no children, we need to continue advancing the iterator until we
    // fall into a node with children (or leaves) to evaluate or we reach the end
    while(isTop_ && isVoidIter() && !isEnd())
      this->operator++();

    return *this;
  }

  bool operator==(const IIRNodeIterator& other) const {
    if(voidIter_)
      return (iterator_ == other.iterator_);

    return (iterator_ == other.iterator_) && (restIterator_ == other.restIterator_);
  }

  bool operator!=(const IIRNodeIterator& other) const { return !(*this == other); }

  void setToEnd() {
    iterator_ = end_;
    if(!voidIter_) {
      // the recursive rest of iterators is constructed with last node (i.e. end-1) and set to end
      // recursively
      restIterator_ =
          IIRNodeIterator<typename RootIIRNode::ChildType, LeafNode>(std::prev(iterator_)->get());
      restIterator_.setToEnd();
    }
  }

  const std::unique_ptr<LeafNode>& operator*() {
    return derefImpl<typename RootIIRNode::ChildType>();
  }

  template <typename T>
  const std::unique_ptr<LeafNode>&
  derefImpl(typename std::enable_if<std::is_same<T, LeafNode>::value>::type* = 0) {
    return *iterator_;
  }

  template <typename T>
  const std::unique_ptr<LeafNode>&
  derefImpl(typename std::enable_if<!std::is_same<T, LeafNode>::value>::type* = 0) {
    return *restIterator_;
  }

  bool isEnd() const {
    if(voidIter_)
      return true;
    return iterator_ == end_;
  }
};

template <typename LeafNode>
class IIRNodeIterator<LeafNode, LeafNode> {
  template <class A, class B>
  friend class IIRNodeIterator;

private:
  IIRNodeIterator() = default;

public:
  IIRNodeIterator(const LeafNode* root) {}
  IIRNodeIterator(IIRNodeIterator&&) = default;
  IIRNodeIterator(const IIRNodeIterator&) = default;

  IIRNodeIterator& operator=(IIRNodeIterator&&) = default;

  IIRNodeIterator& operator++() { return *this; }

  bool isVoidIter() const { return false; }

  bool isEnd() const { return true; }
  void setToEnd() {}
  bool operator==(const IIRNodeIterator& other) const { return true; }
  bool operator!=(const IIRNodeIterator& other) const { return true; }
};

template <typename RootNode, typename LeafNode>
class IIRRange {
  const RootNode& root_;

public:
  IIRRange(const RootNode& root) : root_(root) {}
  IIRNodeIterator<RootNode, LeafNode> begin() {
    auto nodeIter = IIRNodeIterator<RootNode, LeafNode>(&root_, true);
    while(nodeIter.isVoidIter() && !nodeIter.isEnd()) {
      ++nodeIter;
    }
    return nodeIter;
  }
  IIRNodeIterator<RootNode, LeafNode> end() {
    auto it = IIRNodeIterator<RootNode, LeafNode>(&root_, true);
    it.setToEnd();
    return it;
  }
};

} // namespace iir

template <typename Leaf, typename RootNode>
iir::IIRRange<RootNode, Leaf> iterateIIROver(const RootNode& root) {
  return iir::IIRRange<RootNode, Leaf>(root);
}

} // namespace dawn

namespace std {

template <typename RootNode, typename LeafNode>
struct iterator_traits<dawn::iir::IIRNodeIterator<RootNode, LeafNode>> {
  using difference_type =
      typename iterator_traits<typename LeafNode::ChildIterator>::difference_type;
  using value_type = typename iterator_traits<typename LeafNode::ChildIterator>::value_type;
  using pointer = typename iterator_traits<typename LeafNode::ChildIterator>::pointer;
  using iterator_category =
      typename iterator_traits<typename LeafNode::ChildIterator>::iterator_category;
};
} // namespace std
