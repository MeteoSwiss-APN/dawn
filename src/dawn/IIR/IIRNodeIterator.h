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

#ifndef DAWN_IIR_IIRNODEITERATOR_H
#define DAWN_IIR_IIRNODEITERATOR_H

#include "dawn/Support/Assert.h"
#include "dawn/Support/Unreachable.h"
#include <memory>
#include <type_traits>

#include <iterator>
#include <iostream>

namespace dawn {
namespace iir {

template <typename RootIIRNode, typename LeafNode>
class IIRNodeIterator {

  typename RootIIRNode::ChildConstIterator iterator_;
  typename RootIIRNode::ChildConstIterator end_;
  IIRNodeIterator<typename RootIIRNode::ChildType, LeafNode> restIterator_;

public:
  using leafIterator_t = typename LeafNode::ChildIterator;

  IIRNodeIterator(const RootIIRNode* root) {
    DAWN_ASSERT(root);
    DAWN_ASSERT_MSG((root->getChildren().size() > 0),
                    "multi-iterator with empty children set is not supported");
    iterator_ = (root->childrenBegin());
    end_ = (root->childrenEnd());
    restIterator_ =
        IIRNodeIterator<typename RootIIRNode::ChildType, LeafNode>(root->childrenBegin()->get());
  }

  IIRNodeIterator() = default;
  IIRNodeIterator(IIRNodeIterator&&) = default;
  IIRNodeIterator(const IIRNodeIterator&) = default;
  IIRNodeIterator& operator=(IIRNodeIterator&&) = default;

  IIRNodeIterator& operator++() {
    ++restIterator_;
    if(restIterator_.isEnd()) {
      ++iterator_;

      if(!isEnd()) {
        restIterator_ =
            IIRNodeIterator<typename RootIIRNode::ChildType, LeafNode>(iterator_->get());
      }
    }
    return *this;
  }

  bool operator==(const IIRNodeIterator& other) const {
    return (iterator_ == other.iterator_) && (restIterator_ == other.restIterator_);
  }

  bool operator!=(const IIRNodeIterator& other) const { return !(*this == other); }

  void setToEnd() {
    iterator_ = end_;
    // the recursive rest of iterators is constructed with last node (i.e. end-1) and set to end
    // recursively
    restIterator_ =
        IIRNodeIterator<typename RootIIRNode::ChildType, LeafNode>(std::prev(iterator_)->get());
    restIterator_.setToEnd();
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

  bool isEnd() const { return iterator_ == end_; }
};

template <typename LeafNode>
class IIRNodeIterator<LeafNode, LeafNode> {
public:
  IIRNodeIterator() = default;
  IIRNodeIterator(const LeafNode* root) {}
  IIRNodeIterator(IIRNodeIterator&&) = default;
  IIRNodeIterator(const IIRNodeIterator&) = default;

  IIRNodeIterator& operator=(IIRNodeIterator&&) = default;

  IIRNodeIterator& operator++() { return *this; }

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
    return IIRNodeIterator<RootNode, LeafNode>(&root_);
  }
  IIRNodeIterator<RootNode, LeafNode> end() {
    auto it = IIRNodeIterator<RootNode, LeafNode>(&root_);
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
}

#endif
