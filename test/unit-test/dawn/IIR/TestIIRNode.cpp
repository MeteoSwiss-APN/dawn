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
#include "dawn/IIR/IIRNodeIterator.h"
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
  Node4(int val) : val_(val) {}
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

    root_->insertChild(make_unique<impl::Node2>(), root_);
    root_->insertChild(make_unique<impl::Node2>(), root_);

    auto node2_It = root_->childrenBegin();

    (*node2_It)->insertChild(make_unique<impl::Node3>());
    (*node2_It)->insertChild(make_unique<impl::Node3>());
    (*node2_It)->insertChild(make_unique<impl::Node3>());

    auto node3_It = (*node2_It)->childrenBegin();
    (*node3_It)->insertChild(make_unique<impl::Node4>(2));
    (*node3_It)->insertChild(make_unique<impl::Node4>(3));

    node3_It++;

    (*node3_It)->insertChild(make_unique<impl::Node4>(5));
    (*node3_It)->insertChild(make_unique<impl::Node4>(6));

    node3_It++;

    (*node3_It)->insertChild(make_unique<impl::Node4>(8));
    (*node3_It)->insertChild(make_unique<impl::Node4>(9));

    node2_It++;

    (*node2_It)->insertChild(make_unique<impl::Node3>());
    (*node2_It)->insertChild(make_unique<impl::Node3>());
    (*node2_It)->insertChild(make_unique<impl::Node3>());

    node3_It = (*node2_It)->childrenBegin();
    (*node3_It)->insertChild(make_unique<impl::Node4>(12));
    (*node3_It)->insertChild(make_unique<impl::Node4>(13));

    node3_It++;

    (*node3_It)->insertChild(make_unique<impl::Node4>(15));
    (*node3_It)->insertChild(make_unique<impl::Node4>(16));

    node3_It++;

    (*node3_It)->insertChild(make_unique<impl::Node4>(18));
    (*node3_It)->insertChild(make_unique<impl::Node4>(19));
  }

  void TearDown() override {}
};

TEST_F(IIRNode, checkTreeConsistency) { EXPECT_TRUE(root_->checkTreeConsistency()); }

TEST_F(IIRNode, replace) {
  const auto& n2_1 = root_->getChildren()[1];
  const auto& n3_3 = n2_1->getChildren()[0];
  const auto& n4_7 = n3_3->getChildren()[1];

  EXPECT_EQ(n4_7->val_, 13);

  std::unique_ptr<impl::Node4> repl = make_unique<impl::Node4>(-13);
  n3_3->replace(n4_7, repl);

  EXPECT_TRUE(root_->checkTreeConsistency());

  std::array<int, 12> res{{2, 3, 5, 6, 8, 9, 12, -13, 15, 16, 18, 19}};

  unsigned long i = 0;
  for(const auto& it : iterateIIROver<impl::Node4>(*root_)) {
    EXPECT_EQ(it->val_, res[i]);
    ++i;
  }
  EXPECT_EQ(i, 12);
}

// test the insertChild API for regular node
TEST_F(IIRNode, insertChild) {
  const auto& n2_1 = root_->getChildren()[1];
  const auto& n3_3 = n2_1->getChildren()[0];

  // we insert multiple children since the consistency can be broken if the STL vector of children
  // gets reallocated because a resize operation
  for(int i = -1; i > -7; --i) {
    n3_3->insertChild(make_unique<impl::Node4>(i));
  }

  EXPECT_EQ(n3_3->getChildren().size(), 8);
  EXPECT_TRUE(root_->checkTreeConsistency());

  std::array<int, 18> res{{2, 3, 5, 6, 8, 9, 12, 13, -1, -2, -3, -4, -5, -6, 15, 16, 18, 19}};

  unsigned long i = 0;
  for(const auto& it : iterateIIROver<impl::Node4>(*root_)) {
    EXPECT_EQ(it->val_, res[i]);
    ++i;
  }
  EXPECT_EQ(i, 18);
}

// test the insertChild API for the top node
TEST_F(IIRNode, insertChildTopNode) {
  auto newn2 = make_unique<impl::Node2>();
  newn2->insertChild(make_unique<impl::Node3>());
  newn2->insertChild(make_unique<impl::Node3>());
  newn2->insertChild(make_unique<impl::Node3>());
  auto newn3It = (newn2)->childrenBegin();
  (*newn3It)->insertChild(make_unique<impl::Node4>(-9));
  newn3It++;
  (*newn3It)->insertChild(make_unique<impl::Node4>(-11));
  (*newn3It)->insertChild(make_unique<impl::Node4>(-12));
  newn3It++;
  (*newn3It)->insertChild(make_unique<impl::Node4>(-15));
  (*newn3It)->insertChild(make_unique<impl::Node4>(-16));
  (*newn3It)->insertChild(make_unique<impl::Node4>(-17));

  root_->insertChild(newn2, root_);

  EXPECT_TRUE(root_->checkTreeConsistency());
  std::array<int, 18> res{{2, 3, 5, 6, 8, 9, 12, 13, 15, 16, 18, 19, -9, -11, -12, -15, -16, -17}};

  unsigned long i = 0;
  for(const auto& it : iterateIIROver<impl::Node4>(*root_)) {
    EXPECT_EQ(it->val_, res[i]);
    ++i;
  }
  EXPECT_EQ(i, 18);
}

// test the insertChild API for regular node
TEST_F(IIRNode, insertChildren) {

  const auto& n2_1 = root_->getChildren()[1];
  const auto& n3_3 = n2_1->getChildren()[0];

  auto newn3 = make_unique<impl::Node3>();

  newn3->insertChild(make_unique<impl::Node4>(-3));
  newn3->insertChild(make_unique<impl::Node4>(-4));
  newn3->insertChild(make_unique<impl::Node4>(-5));
  newn3->insertChild(make_unique<impl::Node4>(-6));

  n3_3->insertChildren(std::next(n3_3->getChildren().begin()),
                       std::make_move_iterator(newn3->getChildren().begin()),
                       std::make_move_iterator(newn3->getChildren().end()));

  EXPECT_TRUE(root_->checkTreeConsistency());
  std::array<int, 16> res{{2, 3, 5, 6, 8, 9, 12, -3, -4, -5, -6, 13, 15, 16, 18, 19}};

  unsigned long i = 0;
  for(const auto& it : iterateIIROver<impl::Node4>(*root_)) {
    EXPECT_EQ(it->val_, res[i]);
    ++i;
  }
  EXPECT_EQ(i, 16);
}

// test the insertChild API for top node
TEST_F(IIRNode, insertChildrenTopNode) {

  auto newn1 = make_unique<impl::Node1>();
  {
    auto newn2 = make_unique<impl::Node2>();
    auto newn3 = make_unique<impl::Node3>();

    newn3->insertChild(make_unique<impl::Node4>(-3));
    newn3->insertChild(make_unique<impl::Node4>(-4));
    newn2->insertChild(std::move(newn3));
    newn1->insertChild(std::move(newn2), newn1);
  }
  {
    auto newn2 = make_unique<impl::Node2>();
    auto newn3 = make_unique<impl::Node3>();

    newn3->insertChild(make_unique<impl::Node4>(-7));
    newn3->insertChild(make_unique<impl::Node4>(-9));
    newn2->insertChild(std::move(newn3));
    newn1->insertChild(std::move(newn2), newn1);
  }

  root_->insertChildren(std::next(root_->getChildren().begin()),
                        std::make_move_iterator(newn1->getChildren().begin()),
                        std::make_move_iterator(newn1->getChildren().end()), root_);
  EXPECT_TRUE(root_->checkTreeConsistency());
  std::array<int, 16> res{{2, 3, 5, 6, 8, 9, -3, -4, -7, -9, 12, 13, 15, 16, 18, 19}};

  unsigned long i = 0;
  for(const auto& it : iterateIIROver<impl::Node4>(*root_)) {
    EXPECT_EQ(it->val_, res[i]);
    ++i;
  }
  EXPECT_EQ(i, 16);
}
}
