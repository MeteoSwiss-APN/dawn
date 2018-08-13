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

#include "dawn/IIR/IIRNode.h"
#include "dawn/Support/STLExtras.h"
#include <gtest/gtest.h>
#include <string>

using namespace dawn;

namespace impl {

class Node2;
class Node3;
class Node4;

class Node1 : public iir::IIRNode<void, Node1, Node2> {
public:
  static constexpr const char* name = "Node1";
};
class Node2 : public iir::IIRNode<Node1, Node2, Node3> {
public:
  static constexpr const char* name = "Node2";
};
class Node3 : public iir::IIRNode<Node2, Node3, Node4> {
public:
  static constexpr const char* name = "Node3";
};
class Node4 : public iir::IIRNode<Node3, Node4, void> {
public:
  static constexpr const char* name = "Node4";
  Node4(int val) : val_(val){};
  Node4(Node4&& other) : val_(other.val_) {}
  int val_;
};
}

namespace {

class IIRNode : public ::testing::Test {

public:
  std::unique_ptr<impl::Node1> root_;

  IIRNode() : root_(make_unique<impl::Node1>()) {}

  void SetUp() override {

    /* Build the following tree
    // Node1:            root
    //                   /  \______________
    //                  /                  \
    // Node2:          N2                  N2
    //                /|\                  /|\
    //               / | \                / | \
    //              /  |  \              /  |  \
    //             /   |   \            /   |   \
    //            /    |    \          /    |    \
    // Node3:    N3    N3   N3        N3    N3   N3
    //          / \   / \   / \      / \   / \   / \
    // Node4:  2   3 5   6 8   9    12 13 15 16 18 19
    */

    root_->insertChild<std::unique_ptr<impl::Node2>>(make_unique<impl::Node2>(), root_);
    root_->insertChild<std::unique_ptr<impl::Node2>>(make_unique<impl::Node2>(), root_);

    auto node2_It = root_->childrenBegin();

    (*node2_It)->insertChild<std::unique_ptr<impl::Node3>>(make_unique<impl::Node3>());
    (*node2_It)->insertChild<std::unique_ptr<impl::Node3>>(make_unique<impl::Node3>());
    (*node2_It)->insertChild<std::unique_ptr<impl::Node3>>(make_unique<impl::Node3>());

    auto node3_It = (*node2_It)->childrenBegin();
    (*node3_It)->insertChild<std::unique_ptr<impl::Node4>>(make_unique<impl::Node4>(2));
    (*node3_It)->insertChild<std::unique_ptr<impl::Node4>>(make_unique<impl::Node4>(3));

    node3_It++;

    (*node3_It)->insertChild<std::unique_ptr<impl::Node4>>(make_unique<impl::Node4>(5));
    (*node3_It)->insertChild<std::unique_ptr<impl::Node4>>(make_unique<impl::Node4>(6));

    node3_It++;

    (*node3_It)->insertChild<std::unique_ptr<impl::Node4>>(make_unique<impl::Node4>(8));
    (*node3_It)->insertChild<std::unique_ptr<impl::Node4>>(make_unique<impl::Node4>(9));

    node2_It++;

    (*node2_It)->insertChild<std::unique_ptr<impl::Node3>>(make_unique<impl::Node3>());
    (*node2_It)->insertChild<std::unique_ptr<impl::Node3>>(make_unique<impl::Node3>());
    (*node2_It)->insertChild<std::unique_ptr<impl::Node3>>(make_unique<impl::Node3>());

    node3_It = (*node2_It)->childrenBegin();
    (*node3_It)->insertChild<std::unique_ptr<impl::Node4>>(make_unique<impl::Node4>(12));
    (*node3_It)->insertChild<std::unique_ptr<impl::Node4>>(make_unique<impl::Node4>(13));

    node3_It++;

    (*node3_It)->insertChild<std::unique_ptr<impl::Node4>>(make_unique<impl::Node4>(15));
    (*node3_It)->insertChild<std::unique_ptr<impl::Node4>>(make_unique<impl::Node4>(16));

    node3_It++;

    (*node3_It)->insertChild<std::unique_ptr<impl::Node4>>(make_unique<impl::Node4>(18));
    (*node3_It)->insertChild<std::unique_ptr<impl::Node4>>(make_unique<impl::Node4>(19));
  }

  void TearDown() override {}
};

TEST_F(IIRNode, replace) {
  const auto& n2_1 = root_->getChildren()[1];
  const auto& n3_4 = n2_1->getChildren()[0];

  //  const auto& n4_7 = n3_4->getChildren()[1];
  const std::unique_ptr<impl::Node4>& n4_7 = n3_4->getChildren()[1];

  ASSERT_EQ(n4_7->val_, 13);

  std::unique_ptr<impl::Node4> repl = make_unique<impl::Node4>(-13);
  n3_4->replace(n4_7, repl);
}
}
