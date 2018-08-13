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

#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/IIR.h"
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
TEST(IIRNodeIterator, LeafIterator) {

  std::unique_ptr<impl::Node1> root = make_unique<impl::Node1>();

  root->insertChild<std::unique_ptr<impl::Node2>>(make_unique<impl::Node2>(), root);
  root->insertChild<std::unique_ptr<impl::Node2>>(make_unique<impl::Node2>(), root);

  auto node2_It = root->childrenBegin();

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

  iir::IIRNodeIterator<impl::Node1, impl::Node4> fullIt(root.get());

  ASSERT_EQ((*fullIt)->val_, 2);
  ++fullIt;
  ASSERT_EQ((*fullIt)->val_, 3);
  ++fullIt;
  ASSERT_EQ((*fullIt)->val_, 5);
  ++fullIt;
  ASSERT_EQ((*fullIt)->val_, 6);
  ++fullIt;
  ASSERT_EQ((*fullIt)->val_, 8);
  ++fullIt;
  ASSERT_EQ((*fullIt)->val_, 9);
  ++fullIt;
  ASSERT_EQ((*fullIt)->val_, 12);
  ++fullIt;
  ASSERT_EQ((*fullIt)->val_, 13);
  ++fullIt;
  ASSERT_EQ((*fullIt)->val_, 15);
  ++fullIt;
  ASSERT_EQ((*fullIt)->val_, 16);
  ++fullIt;
  ASSERT_EQ((*fullIt)->val_, 18);
  ++fullIt;
  ASSERT_EQ((*fullIt)->val_, 19);

  std::array<int, 12> res{2, 3, 5, 6, 8, 9, 12, 13, 15, 16, 18, 19};
  int i = 0;
  for(const auto& it : iterateIIROver<impl::Node4>(*root)) {
    ASSERT_EQ(it->val_, res[i]);
    ++i;
  }
}
}
