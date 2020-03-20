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

#pragma once

#include <algorithm>
#include <gtest/gtest.h>

#include "atlas/grid.h"
#include "atlas/mesh/actions/BuildEdges.h"
#include "atlas/meshgenerator.h"
#include "atlas/option/Options.h"

#include <atlas/util/CoordinateEnums.h>

#include "interface/atlas_interface.hpp"

namespace {

class TestAtlasInterface : public ::testing::Test {
private:
  atlas::Mesh mesh_;
  int testIdx_;

protected:
  const atlas::Mesh& getMesh() const { return mesh_; }
  int testIdx() const { return testIdx_; }

  explicit TestAtlasInterface() {
    int nx = 10;
    int ny = 10;

    // test idx, found visually. neither on a boundary for edge, node nor cell
    testIdx_ = nx * ny / 2. + ny / 1.5;

    // right handed triangle mesh
    auto x = atlas::grid::LinearSpacing(0, nx, nx, false);
    auto y = atlas::grid::LinearSpacing(0, ny, ny, false);
    atlas::Grid grid = atlas::StructuredGrid{x, y};

    auto meshgen = atlas::StructuredMeshGenerator{atlas::util::Config("angle", -1.)};
    mesh_ = meshgen.generate(grid);

    // coordinate trafo to mold this into an equilat mesh
    auto xy = atlas::array::make_view<double, 2>(mesh_.nodes().xy());
    for(int nodeIdx = 0; nodeIdx < mesh_.nodes().size(); nodeIdx++) {
      double x = xy(nodeIdx, atlas::LON);
      double y = xy(nodeIdx, atlas::LAT);
      x = x - 0.5 * y;
      y = y * sqrt(3) / 2.;
      xy(nodeIdx, atlas::LON) = x;
      xy(nodeIdx, atlas::LAT) = y;
    }

    // build up nbh tables
    atlas::mesh::actions::build_edges(mesh_, atlas::util::Config("pole_edges", false));
    atlas::mesh::actions::build_node_to_edge_connectivity(mesh_);
    atlas::mesh::actions::build_element_to_edge_connectivity(mesh_);

    // mesh constructed this way is missing node to cell connectivity, built it as well
    for(int nodeIdx = 0; nodeIdx < mesh_.nodes().size(); nodeIdx++) {
      const auto& nodeToEdge = mesh_.nodes().edge_connectivity();
      const auto& edgeToCell = mesh_.edges().cell_connectivity();
      auto& nodeToCell = mesh_.nodes().cell_connectivity();

      std::set<int> nbh;
      for(int nbhEdgeIdx = 0; nbhEdgeIdx < nodeToEdge.cols(nodeIdx); nbhEdgeIdx++) {
        int edgeIdx = nodeToEdge(nodeIdx, nbhEdgeIdx);
        if(edgeIdx == nodeToEdge.missing_value()) {
          continue;
        }
        for(int nbhCellIdx = 0; nbhCellIdx < edgeToCell.cols(edgeIdx); nbhCellIdx++) {
          int cellIdx = edgeToCell(edgeIdx, nbhCellIdx);
          if(cellIdx == edgeToCell.missing_value()) {
            continue;
          }
          nbh.insert(cellIdx);
        }
      }

      assert(nbh.size() <= 6);
      std::vector<int> initData(nbh.size(), nodeToCell.missing_value());
      nodeToCell.add(1, nbh.size(), initData.data());
      int copyIter = 0;
      for(const int n : nbh) {
        nodeToCell.set(nodeIdx, copyIter++, n);
      }
    }
  }
};

// compare two (partial neighborhoods)
bool nbhsValidAndEqual(std::vector<int> a, std::vector<int> b) {
  // need to have same length
  if(!(a.size() == b.size())) {
    return false;
  }

  // same elements, but order may be different
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  if(!(a == b)) {
    return false;
  }

  // unique elements only
  auto isUnique = [](std::vector<int> in) -> bool {
    auto it = std::unique(in.begin(), in.end());
    return (it == in.end());
  };

  return (isUnique(a) && isUnique(b));
}

TEST_F(TestAtlasInterface, Diamond) {
  std::vector<dawn::LocationType> chain{dawn::LocationType::Edges, dawn::LocationType::Cells,
                                        dawn::LocationType::Vertices};
  std::vector<int> diamond = atlasInterface::getNeighbors(getMesh(), chain, testIdx());

  ASSERT_TRUE(diamond.size() == 4);
  std::vector<int> diamondLo{diamond[0], diamond[1]};
  std::vector<int> diamondHi{diamond[2], diamond[3]};

  std::vector<int> diamondLoRef = {getMesh().edges().node_connectivity()(testIdx(), 0),
                                   getMesh().edges().node_connectivity()(testIdx(), 1)};
  ASSERT_TRUE(nbhsValidAndEqual(diamondLoRef, diamondLo));

  std::vector<int> diamondHiRef;
  const int nodesPerCell = 3;
  int cIdx0 = getMesh().edges().cell_connectivity()(testIdx(), 0);
  int cIdx1 = getMesh().edges().cell_connectivity()(testIdx(), 1);
  for(int nbhIdx = 0; nbhIdx < nodesPerCell; nbhIdx++) {
    int vIdx0 = getMesh().cells().node_connectivity()(cIdx0, nbhIdx);
    int vIdx1 = getMesh().cells().node_connectivity()(cIdx1, nbhIdx);
    if(std::find(diamondHiRef.begin(), diamondHiRef.end(), vIdx0) == diamondHiRef.end() &&
       std::find(diamondLoRef.begin(), diamondLoRef.end(), vIdx0) == diamondLoRef.end()) {
      diamondHiRef.push_back(vIdx0);
    }
    if(std::find(diamondHiRef.begin(), diamondHiRef.end(), vIdx1) == diamondHiRef.end() &&
       std::find(diamondLoRef.begin(), diamondLoRef.end(), vIdx1) == diamondLoRef.end()) {
      diamondHiRef.push_back(vIdx1);
    }
  }

  ASSERT_TRUE(nbhsValidAndEqual(diamondHiRef, diamondHi));
}

TEST_F(TestAtlasInterface, Star) {
  std::vector<dawn::LocationType> chain{
      dawn::LocationType::Vertices,
      dawn::LocationType::Cells,
      dawn::LocationType::Edges,
      dawn::LocationType::Cells,
  };
  std::vector<int> star = atlasInterface::getNeighbors(getMesh(), chain, testIdx());
  ASSERT_TRUE(star.size() == 12);
  std::vector<int> starLo{star.begin(), star.begin() + 6};
  std::vector<int> starHi{star.begin() + 6, star.end()};

  std::vector<int> starLoRef;
  std::vector<int> starHiRef;

  const int cellsPerVertex = 6;
  for(int nbhIter = 0; nbhIter < cellsPerVertex; nbhIter++) {
    starLoRef.push_back(getMesh().nodes().cell_connectivity()(testIdx(), nbhIter));
  }

  ASSERT_TRUE(nbhsValidAndEqual(starLoRef, starLo));

  const int edgesPerCell = 3;
  for(int cellLo : starLoRef) {
    for(int nbhEdge = 0; nbhEdge < edgesPerCell; nbhEdge++) {
      int edgeIdx = getMesh().cells().edge_connectivity()(cellLo, nbhEdge);
      int c0 = getMesh().edges().cell_connectivity()(edgeIdx, 0);
      int c1 = getMesh().edges().cell_connectivity()(edgeIdx, 1);
      if(std::find(starHiRef.begin(), starHiRef.end(), c0) == starHiRef.end() &&
         std::find(starLoRef.begin(), starLoRef.end(), c0) == starLoRef.end()) {
        starHiRef.push_back(c0);
      }
      if(std::find(starHiRef.begin(), starHiRef.end(), c1) == starHiRef.end() &&
         std::find(starLoRef.begin(), starLoRef.end(), c1) == starLoRef.end()) {
        starHiRef.push_back(c1);
      }
    }
  }

  ASSERT_TRUE(nbhsValidAndEqual(starHiRef, starHi));
}

TEST_F(TestAtlasInterface, Fan) {
  std::vector<dawn::LocationType> chain{
      dawn::LocationType::Vertices,
      dawn::LocationType::Cells,
      dawn::LocationType::Edges,
  };
  std::vector<int> fan = atlasInterface::getNeighbors(getMesh(), chain, testIdx());
  ASSERT_TRUE(fan.size() == 12);
  std::vector<int> fanLo{fan.begin(), fan.begin() + 6};
  std::vector<int> fanHi{fan.begin() + 6, fan.end()};

  std::vector<int> fanLoRef;
  std::vector<int> fanHiRef;

  const int edgesPerVertex = 6;
  for(int nbhIter = 0; nbhIter < edgesPerVertex; nbhIter++) {
    fanLoRef.push_back(getMesh().nodes().edge_connectivity()(testIdx(), nbhIter));
  }

  ASSERT_TRUE(nbhsValidAndEqual(fanLoRef, fanLo));
  const int cellsPerVertex = 6;
  const int edgesPerCell = 3;
  for(int nbhIter = 0; nbhIter < cellsPerVertex; nbhIter++) {
    int cIdx = getMesh().nodes().cell_connectivity()(testIdx(), nbhIter);
    for(int edgeNbh = 0; edgeNbh < edgesPerCell; edgeNbh++) {
      int eIdx = getMesh().cells().edge_connectivity()(cIdx, edgeNbh);
      if(std::find(fanHiRef.begin(), fanHiRef.end(), eIdx) == fanHiRef.end() &&
         std::find(fanLoRef.begin(), fanLoRef.end(), eIdx) == fanLoRef.end()) {
        fanHiRef.push_back(eIdx);
      }
    }
  }
  ASSERT_TRUE(nbhsValidAndEqual(fanHiRef, fanHi));
}

TEST_F(TestAtlasInterface, Intp) {
  std::vector<dawn::LocationType> chain{dawn::LocationType::Cells, dawn::LocationType::Edges,
                                        dawn::LocationType::Cells, dawn::LocationType::Edges,
                                        dawn::LocationType::Cells};
  std::vector<int> intp = atlasInterface::getNeighbors(getMesh(), chain, testIdx());
  ASSERT_TRUE(intp.size() == 9);
  std::vector<int> intpLo{intp.begin(), intp.begin() + 3};
  std::vector<int> intpHi{intp.begin() + 3, intp.end()};

  std::vector<int> intpLoRef;
  std::vector<int> intpHiRef;

  const int edgesPerCell = 3;
  for(int nbhIter = 0; nbhIter < edgesPerCell; nbhIter++) {
    int edgeIdx = getMesh().cells().edge_connectivity()(testIdx(), nbhIter);
    intpLoRef.push_back(getMesh().edges().cell_connectivity()(edgeIdx, 0) == testIdx()
                            ? getMesh().edges().cell_connectivity()(edgeIdx, 1)
                            : getMesh().edges().cell_connectivity()(edgeIdx, 0));
  }
  ASSERT_TRUE(nbhsValidAndEqual(intpLoRef, intpLo));

  for(int cIdx : intpLo) {
    for(int nbhIter = 0; nbhIter < edgesPerCell; nbhIter++) {
      int edgeIdx = getMesh().cells().edge_connectivity()(cIdx, nbhIter);
      int nbhC0 = getMesh().edges().cell_connectivity()(edgeIdx, 0);
      int nbhC1 = getMesh().edges().cell_connectivity()(edgeIdx, 1);

      if(nbhC0 != cIdx && nbhC0 != testIdx()) {
        intpHiRef.push_back(nbhC0);
      }
      if(nbhC1 != cIdx && nbhC1 != testIdx()) {
        intpHiRef.push_back(nbhC1);
      }
    }
  }

  ASSERT_TRUE(nbhsValidAndEqual(intpHiRef, intpHi));
}
} // namespace