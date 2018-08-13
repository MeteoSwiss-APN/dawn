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

/// @brief Convencience graph to test the is-DAG algorithm
class TestGraph : public iir::DependencyGraphAccesses {
  using Base = iir::DependencyGraphAccesses;

public:
  TestGraph() : Base(nullptr) {}
  void insertEdge(int IDFrom, int IDTo) {
    Base::insertNode(IDFrom);
    Base::insertEdge(IDFrom, IDTo, Extents{0, 0, 0, 0, 0, 0});
  }
};

TEST(IsDAGAlgorithmTest, Test1) {
  TestGraph graph;

  graph.insertEdge(0, 1);

  EXPECT_TRUE(graph.isDAG());
}

TEST(IsDAGAlgorithmTest, Test2) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 0);

  EXPECT_FALSE(graph.isDAG());
}

TEST(IsDAGAlgorithmTest, Test3) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 0);
  graph.insertEdge(2, 1);

  EXPECT_FALSE(graph.isDAG());
}

TEST(IsDAGAlgorithmTest, Test4) {
  TestGraph graph;

  // Internal cycles are ok!
  graph.insertEdge(0, 1);
  graph.insertEdge(1, 2);
  graph.insertEdge(2, 1);
  graph.insertEdge(2, 3);

  EXPECT_TRUE(graph.isDAG());
}

TEST(IsDAGAlgorithmTest, Test5) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 2);
  graph.insertEdge(2, 1);
  graph.insertEdge(2, 3);

  graph.insertEdge(4, 5);

  EXPECT_TRUE(graph.isDAG());
}

TEST(IsDAGAlgorithmTest, Test6) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 2);
  graph.insertEdge(2, 1);
  graph.insertEdge(2, 3);

  graph.insertEdge(4, 5);
  graph.insertEdge(5, 4);

  EXPECT_FALSE(graph.isDAG());
}

TEST(IsDAGAlgorithmTest, Test7) {
  TestGraph graph;

  // Self-dependencies are fine!
  graph.insertEdge(0, 0);

  EXPECT_TRUE(graph.isDAG());
}

} // anonymous namespace
