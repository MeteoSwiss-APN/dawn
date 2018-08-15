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

#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/Support/STLExtras.h"
#include <gtest/gtest.h>
#include <set>

using namespace dawn;

namespace {

/// @brief Convencience graph to test the strongly connected components algorithm
class TestGraph : public iir::DependencyGraphAccesses {
  using Base = iir::DependencyGraphAccesses;

public:
  TestGraph() : Base(nullptr) {}
  void insertEdge(int IDFrom, int IDTo) {
    Base::insertNode(IDFrom);
    Base::insertEdge(IDFrom, IDTo, iir::Extents{0, 0, 0, 0, 0, 0});
  }
};

std::vector<std::set<int>> makeSCC() { return std::vector<std::set<int>>(); }

TEST(SCCAlgorithmTest, Test1) {
  TestGraph graph;

  graph.insertEdge(0, 1);

  ASSERT_FALSE(graph.hasStronglyConnectedComponents());

  auto scc = makeSCC();
  ASSERT_FALSE(graph.findStronglyConnectedComponents(scc));
  EXPECT_TRUE(scc.empty());
}

TEST(SCCAlgorithmTest, Test2) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 2);

  ASSERT_FALSE(graph.hasStronglyConnectedComponents());
}

TEST(SCCAlgorithmTest, Test3) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 1);

  ASSERT_FALSE(graph.hasStronglyConnectedComponents());
}

TEST(SCCAlgorithmTest, Test4) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 0);

  auto scc = makeSCC();
  ASSERT_TRUE(graph.findStronglyConnectedComponents(scc));
  EXPECT_TRUE((scc[0] == std::set<int>{0, 1}));
}

TEST(SCCAlgorithmTest, Test5) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 2);
  graph.insertEdge(2, 0);

  auto scc = makeSCC();
  ASSERT_TRUE(graph.findStronglyConnectedComponents(scc));
  EXPECT_TRUE((scc[0] == std::set<int>{0, 1, 2}));
}

TEST(SCCAlgorithmTest, Test6) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(2, 1);
  graph.insertEdge(3, 1);

  ASSERT_FALSE(graph.hasStronglyConnectedComponents());
}

TEST(SCCAlgorithmTest, Test7) {
  TestGraph graph;

  /*
                                 +- - - - - - - - - -+
                                 v                   '
           +---+     +---+     +---+     +---+     +---+
        +  | 0 | - > | 1 | - > | 2 | - > | 3 | - > | 4 |
        '  +---+     +---+     +---+     +---+     +---+
        '
        '    +- - - - -+
        '    v         '
        '  +---+     +---+
        +> | 5 | - > | 6 |
           +---+     +---+
  */

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 2);

  // SCC: 1
  graph.insertEdge(2, 3);
  graph.insertEdge(3, 4);
  graph.insertEdge(4, 2);

  // SCC: 2
  graph.insertEdge(0, 5);
  graph.insertEdge(5, 6);
  graph.insertEdge(6, 5);

  auto scc = makeSCC();
  ASSERT_TRUE(graph.findStronglyConnectedComponents(scc));
  ASSERT_EQ(scc.size(), 2);

  std::sort(scc.begin(), scc.end(),
            [](const std::set<int>& a, const std::set<int>& b) { return a.size() < b.size(); });

  // SCC: 1
  ASSERT_EQ(scc[0].size(), 2);
  EXPECT_TRUE((scc[0] == std::set<int>{5, 6}));

  // SCC: 2
  ASSERT_EQ(scc[1].size(), 3);
  EXPECT_TRUE((scc[1] == std::set<int>{2, 3, 4}));
}

TEST(SCCAlgorithmTest, Test8) {
  TestGraph graph;

  /*
      + - - - - - - - - - - - - - - +
      v                             '
    +---+     +---+     +---+     +---+
    | 0 | - > | 1 | - > | 2 | - > | 3 |
    +---+     +---+     +---+     +---+

      + - - - - - - - - - +
      v                   '
    +---+     +---+     +---+
    | 5 | - > | 6 | - > | 7 |
    +---+     +---+     +---+

  */

  // SCC: 1
  graph.insertEdge(0, 1);
  graph.insertEdge(1, 2);
  graph.insertEdge(2, 3);
  graph.insertEdge(3, 0);

  // SCC: 2
  graph.insertEdge(5, 6);
  graph.insertEdge(6, 7);
  graph.insertEdge(7, 5);

  auto scc = makeSCC();
  ASSERT_TRUE(graph.findStronglyConnectedComponents(scc));
  ASSERT_EQ(scc.size(), 2);

  std::sort(scc.begin(), scc.end(),
            [](const std::set<int>& a, const std::set<int>& b) { return a.size() < b.size(); });

  // SCC: 1
  ASSERT_EQ(scc[0].size(), 3);
  EXPECT_TRUE((scc[0] == std::set<int>{5, 6, 7}));

  // SCC: 2
  ASSERT_EQ(scc[1].size(), 4);
  EXPECT_TRUE((scc[1] == std::set<int>{0, 1, 2, 3}));
}

TEST(SCCAlgorithmTest, Test9) {
  TestGraph graph;

  /*
                     +- - - - - - - +
                     '              '
                     '              '
           +- - - - -+- - - - -+    '
           '         v         v    '
         +---+     +---+     +---+  '
      +> | 0 | - > | 1 | - > | 2 |  +
      '  +---+     +---+     +---+
      '    ^         '         '
      '    +- - - - -+         '
      '                        '
      '                        '
      + - - - - - - - - - - - -+
  */

  // Clique
  graph.insertEdge(0, 1);
  graph.insertEdge(0, 2);
  graph.insertEdge(1, 0);
  graph.insertEdge(1, 2);
  graph.insertEdge(2, 0);
  graph.insertEdge(2, 1);

  auto scc = makeSCC();
  ASSERT_TRUE(graph.findStronglyConnectedComponents(scc));
  EXPECT_TRUE((scc[0] == std::set<int>{0, 1, 2}));
}

} // anonymous namespace
