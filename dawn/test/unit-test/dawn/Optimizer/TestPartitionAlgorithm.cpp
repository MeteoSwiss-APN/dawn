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

/// @brief Convencience graph to test the partitioning algorithm
class TestGraph : public iir::DependencyGraphAccesses {
  using Base = iir::DependencyGraphAccesses;

public:
  TestGraph() : Base(iir::StencilMetaInformation{sir::GlobalVariableMap{}}) {}
  void insertEdge(int IDFrom, int IDTo) {
    Base::insertNode(IDFrom);
    Base::insertEdge(IDFrom, IDTo, iir::Extents(dawn::ast::cartesian_{}, 0, 0, 0, 0, 0, 0));
  }
};

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

} // anonymous namespace
