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
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Support/STLExtras.h"
#include <gtest/gtest.h>
#include <set>

using namespace dawn;

namespace {

class TestGraph : public iir::DependencyGraphAccesses {
  using Base = iir::DependencyGraphAccesses;

public:
  TestGraph() : Base(iir::StencilMetaInformation{sir::GlobalVariableMap{}}) {}
  void insertEdge(int IDFrom, int IDTo) {
    Base::insertNode(IDFrom);
    Base::insertEdge(IDFrom, IDTo, iir::Extents(dawn::ast::cartesian_{}, 0, 0, 0, 0, 0, 0));
  }
};

TEST(GraphTest, InputOutputNodes1) {
  TestGraph graph;
  graph.insertEdge(0, 1);

  auto outputNodes = graph.getOutputVertexIDs();
  auto inputNodes = graph.getInputVertexIDs();

  ASSERT_EQ(outputNodes.size(), 1);
  EXPECT_TRUE((std::set<std::size_t>(outputNodes.begin(), outputNodes.end()) ==
               std::set<std::size_t>{graph.getVertexIDFromValue(0)}));

  ASSERT_EQ(inputNodes.size(), 1);
  EXPECT_TRUE((std::set<std::size_t>(inputNodes.begin(), inputNodes.end()) ==
               std::set<std::size_t>{graph.getVertexIDFromValue(1)}));
}

TEST(GraphTest, InputOutputNodes2) {
  TestGraph graph;

  /*
           +---+     +---+     +---+     +---+     +---+
        +  | 0 | - > | 1 | - > | 2 | - > | 3 | - > | 4 |
        '  +---+     +---+     +---+     +---+     +---+
        '
        '
        '
        '  +---+     +---+
        +> | 5 | - > | 6 |
           +---+     +---+
  */

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 2);
  graph.insertEdge(2, 3);
  graph.insertEdge(3, 4);

  graph.insertEdge(0, 5);
  graph.insertEdge(5, 6);

  auto outputNodes = graph.getOutputVertexIDs();
  auto inputNodes = graph.getInputVertexIDs();

  ASSERT_EQ(outputNodes.size(), 1);
  EXPECT_TRUE((std::set<std::size_t>(outputNodes.begin(), outputNodes.end()) ==
               std::set<std::size_t>{graph.getVertexIDFromValue(0)}));

  ASSERT_EQ(inputNodes.size(), 2);
  EXPECT_TRUE(
      (std::set<std::size_t>(inputNodes.begin(), inputNodes.end()) ==
       std::set<std::size_t>{graph.getVertexIDFromValue(4), graph.getVertexIDFromValue(6)}));
}

TEST(GraphTest, InputOutputNodes3) {
  TestGraph graph;

  /*
           +---+     +---+
        +  | 0 | < - | 1 |<+
        '  +---+     +---+ '
        '                  '
        '                  '
        '                  '
        '  +---+     +---+ '
        +> | 2 | - > | 3 | +
           +---+     +---+
  */

  graph.insertEdge(0, 2);
  graph.insertEdge(2, 3);
  graph.insertEdge(3, 1);
  graph.insertEdge(1, 0);

  ASSERT_TRUE(graph.getOutputVertexIDs().empty());
  ASSERT_TRUE(graph.getInputVertexIDs().empty());
}

TEST(GraphTest, cycle) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 2);
  graph.insertEdge(2, 3);
  graph.insertEdge(3, 4);

  graph.insertEdge(0, 5);
  graph.insertEdge(5, 6);
  graph.insertEdge(1, 7);
  graph.insertEdge(7, 0);

  EXPECT_TRUE((graph.hasCycleDependency(0)));
  EXPECT_TRUE((graph.hasCycleDependency(1)));
  EXPECT_TRUE((!graph.hasCycleDependency(2)));
  EXPECT_TRUE((!graph.hasCycleDependency(3)));
  EXPECT_TRUE((!graph.hasCycleDependency(4)));
  EXPECT_TRUE((!graph.hasCycleDependency(5)));
  EXPECT_TRUE((!graph.hasCycleDependency(6)));
  EXPECT_TRUE((graph.hasCycleDependency(7)));
}

TEST(GraphTest, computeIDsWithCycle) {
  TestGraph graph;

  graph.insertEdge(3, 5);
  graph.insertEdge(5, 6);
  graph.insertEdge(6, 7);
  graph.insertEdge(7, 8);

  graph.insertEdge(3, 9);
  graph.insertEdge(9, 10);
  graph.insertEdge(5, 11);
  graph.insertEdge(11, 3);

  auto ids = graph.computeIDsWithCycles();
  std::vector<int> ref = {3, 5, 11};
  EXPECT_TRUE((std::equal(ids.begin(), ids.end(), ref.begin())));
}

} // anonymous namespace
