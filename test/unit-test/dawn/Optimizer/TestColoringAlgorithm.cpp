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
#include <algorithm>
#include <numeric>

using namespace dawn;

namespace {

/// @brief Convencience graph to test the coloring algorithm
class TestGraph : public iir::DependencyGraphAccesses {
  using Base = iir::DependencyGraphAccesses;

public:
  TestGraph() : Base(nullptr) {}
  void insertEdge(int IDFrom, int IDTo) {
    Base::insertNode(IDFrom);
    Base::insertEdge(IDFrom, IDTo, iir::Extents{0, 0, 0, 0, 0, 0});
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

} // anonymous namespace
