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
#include "dawn/SIR/SIR.h"
#include "dawn/Support/STLExtras.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <set>

using namespace dawn;

namespace {

/// @brief Convencience graph to test the coloring algorithm
class TestGraph : public iir::DependencyGraphAccesses {
  using Base = iir::DependencyGraphAccesses;

public:
  TestGraph() : Base(iir::StencilMetaInformation{std::make_shared<sir::GlobalVariableMap>()}) {}
  void insertEdge(int IDFrom, int IDTo) {
    Base::insertNode(IDFrom);
    Base::insertEdge(IDFrom, IDTo, iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0));
  }
};

int getNumColors(const std::unordered_map<int, int>& coloring) {
  std::set<int> colors;
  return std::accumulate(coloring.begin(), coloring.end(), 0,
                         [&](int numColors, const std::pair<int, int>& AccessIDColorPair) {
                           int color = AccessIDColorPair.second;
                           return colors.insert(color).second ? numColors + 1 : numColors;
                         });
}

//===------------------------------------------------------------------------------------------===//
// Coloring Algorithm
//===------------------------------------------------------------------------------------------===//

TEST(ColoringAlgorithmTest, Test1) {
  TestGraph graph;
  graph.insertEdge(0, 1);

  std::unordered_map<int, int> coloring;
  graph.greedyColoring(coloring);

  EXPECT_EQ(getNumColors(coloring), 2);
}

TEST(ColoringAlgorithmTest, Test2) {
  TestGraph graph;

  // Clique
  graph.insertEdge(0, 1);
  graph.insertEdge(0, 2);
  graph.insertEdge(1, 0);
  graph.insertEdge(1, 2);
  graph.insertEdge(2, 0);
  graph.insertEdge(2, 1);

  std::unordered_map<int, int> coloring;
  graph.greedyColoring(coloring);

  EXPECT_EQ(getNumColors(coloring), 3);
}

TEST(ColoringAlgorithmTest, Test3) {
  TestGraph graph;

  // Star
  graph.insertEdge(0, 1);
  graph.insertEdge(0, 2);
  graph.insertEdge(0, 3);
  graph.insertEdge(0, 4);
  graph.insertEdge(0, 5);
  graph.insertEdge(0, 6);

  std::unordered_map<int, int> coloring;
  graph.greedyColoring(coloring);

  EXPECT_EQ(getNumColors(coloring), 2);
}

TEST(ColoringAlgorithmTest, Test4) {
  TestGraph graph;
  graph.insertEdge(0, 1);
  graph.insertEdge(1, 2);
  graph.insertEdge(2, 3);
  graph.insertEdge(3, 4);

  std::unordered_map<int, int> coloring;
  graph.greedyColoring(coloring);

  EXPECT_EQ(getNumColors(coloring), 2);
}

TEST(ColoringAlgorithmTest, Test5) {
  TestGraph graph;
  graph.insertEdge(0, 2);
  graph.insertEdge(1, 2);
  graph.insertEdge(2, 3);
  graph.insertEdge(3, 4);
  graph.insertEdge(4, 5);

  std::unordered_map<int, int> coloring;
  graph.greedyColoring(coloring);

  EXPECT_EQ(getNumColors(coloring), 2);
}

TEST(ColoringAlgorithmTest, Test6) {
  TestGraph graph;
  graph.insertNode(0);
  graph.insertNode(1);
  graph.insertNode(2);
  graph.insertNode(3);

  std::unordered_map<int, int> coloring;
  graph.greedyColoring(coloring);

  EXPECT_EQ(coloring.size(), 4);
  EXPECT_EQ(getNumColors(coloring), 1);
}

//===------------------------------------------------------------------------------------------===//
// Graph
//===------------------------------------------------------------------------------------------===//

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

//===------------------------------------------------------------------------------------------===//
// Is DAG
//===------------------------------------------------------------------------------------------===//

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

//===------------------------------------------------------------------------------------------===//
// Partition Algorithm
//===------------------------------------------------------------------------------------------===//

/// @brief Check that the partition given by `IDs` is in `partitions`
template <class... IDTypes>
bool partitionIsIn(TestGraph& graph, std::vector<std::set<std::size_t>> partitions,
                   IDTypes... IDs) {
  for(const auto& partition : partitions)
    if(partition == std::set<std::size_t>{graph.getVertexIDFromValue(IDs)...})
      return true;
  return false;
}

TEST(PartitionAlgorithmTest, Test1) {
  TestGraph graph;

  graph.insertEdge(0, 1);

  std::vector<std::set<std::size_t>> partitions = graph.partitionInSubGraphs();
  EXPECT_EQ(partitions.size(), 1);
  EXPECT_TRUE(partitionIsIn(graph, partitions, 0, 1));
}

TEST(PartitionAlgorithmTest, Test2) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 0);

  std::vector<std::set<std::size_t>> partitions = graph.partitionInSubGraphs();
  EXPECT_EQ(partitions.size(), 1);
  EXPECT_TRUE(partitionIsIn(graph, partitions, 0, 1));
}

TEST(PartitionAlgorithmTest, Test3) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(0, 2);

  std::vector<std::set<std::size_t>> partitions = graph.partitionInSubGraphs();
  EXPECT_EQ(partitions.size(), 1);
  EXPECT_TRUE(partitionIsIn(graph, partitions, 0, 1, 2));
}

TEST(PartitionAlgorithmTest, Test4) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(2, 3);

  std::vector<std::set<std::size_t>> partitions = graph.partitionInSubGraphs();
  EXPECT_EQ(partitions.size(), 2);
  EXPECT_TRUE(partitionIsIn(graph, partitions, 0, 1));
  EXPECT_TRUE(partitionIsIn(graph, partitions, 2, 3));
}

TEST(PartitionAlgorithmTest, Test5) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 1);

  graph.insertEdge(4, 5);
  graph.insertEdge(4, 4);

  graph.insertEdge(2, 3);
  graph.insertEdge(2, 2);

  std::vector<std::set<std::size_t>> partitions = graph.partitionInSubGraphs();
  EXPECT_EQ(partitions.size(), 3);
  EXPECT_TRUE(partitionIsIn(graph, partitions, 0, 1));
  EXPECT_TRUE(partitionIsIn(graph, partitions, 2, 3));
  EXPECT_TRUE(partitionIsIn(graph, partitions, 4, 5));
}

TEST(PartitionAlgorithmTest, Test6) {
  TestGraph graph;

  graph.insertEdge(0, 1);
  graph.insertEdge(1, 1);

  graph.insertEdge(4, 5);
  graph.insertEdge(4, 4);

  graph.insertEdge(2, 3);
  graph.insertEdge(2, 2);

  graph.insertEdge(2, 4);

  std::vector<std::set<std::size_t>> partitions = graph.partitionInSubGraphs();
  EXPECT_EQ(partitions.size(), 2);
  EXPECT_TRUE(partitionIsIn(graph, partitions, 0, 1));
  EXPECT_TRUE(partitionIsIn(graph, partitions, 2, 3, 4, 5));
}

//===------------------------------------------------------------------------------------------===//
// Strongly Connected Components
//===------------------------------------------------------------------------------------------===//

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
