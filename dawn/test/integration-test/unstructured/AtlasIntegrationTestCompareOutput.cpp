//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "AtlasCartesianWrapper.h"
#include "UnstructuredVerifier.h"

#include "atlas/grid.h"
#include "atlas/mesh/actions/BuildCellCentres.h"
#include "atlas/mesh/actions/BuildEdges.h"
#include "atlas/mesh/actions/BuildPeriodicBoundaries.h"
#include "atlas/meshgenerator.h"
#include "atlas/option/Options.h"
#include "atlas/output/Gmsh.h"
#include "interface/atlas_interface.hpp"

#include <atlas/util/CoordinateEnums.h>

#include <gtest/gtest.h>

#include <sstream>
#include <tuple>

#include <generated_copyCell.hpp>
namespace {

atlas::Mesh generateQuadMesh(size_t nx, size_t ny) {
  std::stringstream configStr;
  configStr << "L" << nx << "x" << ny;
  atlas::StructuredGrid structuredGrid = atlas::Grid(configStr.str());
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);
  atlas::mesh::actions::build_edges(
      mesh, atlas::util::Config("pole_edges", false)); // work around to eliminate pole edges
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
  atlas::mesh::actions::build_element_to_edge_connectivity(mesh);
  return mesh;
}

atlas::Mesh generateEquilatMesh(size_t nx, size_t ny) {

  // right handed triangle mesh
  auto x = atlas::grid::LinearSpacing(0, nx, nx, false);
  auto y = atlas::grid::LinearSpacing(0, ny, ny, false);
  atlas::Grid grid = atlas::StructuredGrid{x, y};

  auto meshgen = atlas::StructuredMeshGenerator{atlas::util::Config("angle", -1.)};
  atlas::Mesh mesh = meshgen.generate(grid);

  // coordinate trafo to mold this into an equilat mesh
  auto xy = atlas::array::make_view<double, 2>(mesh.nodes().xy());
  for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
    double x = xy(nodeIdx, atlas::LON);
    double y = xy(nodeIdx, atlas::LAT);
    x = x - 0.5 * y;
    y = y * sqrt(3) / 2.;
    xy(nodeIdx, atlas::LON) = x;
    xy(nodeIdx, atlas::LAT) = y;
  }

  // build up nbh tables
  atlas::mesh::actions::build_edges(mesh, atlas::util::Config("pole_edges", false));
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
  atlas::mesh::actions::build_element_to_edge_connectivity(mesh);

  // mesh constructed this way is missing node to cell connectivity, built it as well
  for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
    const auto& nodeToEdge = mesh.nodes().edge_connectivity();
    const auto& edgeToCell = mesh.edges().cell_connectivity();
    auto& nodeToCell = mesh.nodes().cell_connectivity();

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

  return mesh;
}

std::tuple<atlas::Field, atlasInterface::Field<double>> makeAtlasField(const std::string& name,
                                                                       size_t size, size_t kSize) {
  atlas::Field field_F{name, atlas::array::DataType::real64(),
                       atlas::array::make_shape(size, kSize)};
  return {field_F, atlas::array::make_view<double, 2>(field_F)};
}

std::tuple<atlas::Field, atlasInterface::SparseDimension<double>>
makeAtlasSparseField(const std::string& name, size_t size, size_t sparseSize, int kSize) {
  atlas::Field field_F{name, atlas::array::DataType::real64(),
                       atlas::array::make_shape(size, kSize, sparseSize)};
  return {field_F, atlas::array::make_view<double, 3>(field_F)};
}

template <typename T>
void initField(atlasInterface::Field<T>& field, size_t numEl, size_t kSize, T val) {
  for(int level = 0; level < kSize; ++level) {
    for(int elIdx = 0; elIdx < numEl; ++elIdx) {
      field(elIdx, level) = val;
    }
  }
}

template <typename T>
void initSparseField(atlasInterface::SparseDimension<T>& sparseField, size_t numEl, size_t kSize,
                     size_t sparseSize, T val) {
  for(int level = 0; level < kSize; ++level) {
    for(int elIdx = 0; elIdx < numEl; ++elIdx) {
      for(int nbhIdx = 0; nbhIdx < sparseSize; nbhIdx++) {
        sparseField(elIdx, nbhIdx, level) = val;
      }
    }
  }
}

namespace {
#include <generated_copyCell.hpp>
TEST(AtlasIntegrationTestCompareOutput, CopyCell) {
  // Setup a 32 by 32 grid of quads and generate a mesh out of it
  auto mesh = generateQuadMesh(32, 32);
  // We only need one vertical level
  size_t nb_levels = 1;

  auto [in_F, in_v] = makeAtlasField("in", mesh.cells().size(), nb_levels);
  auto [out_F, out_v] = makeAtlasField("out", mesh.cells().size(), nb_levels);

  // Initialize fields with data
  initField(in_v, mesh.cells().size(), nb_levels, 1.0);
  initField(out_v, mesh.cells().size(), nb_levels, -1.0);

  // Run the stencil
  dawn_generated::cxxnaiveico::copyCell<atlasInterface::atlasTag>(mesh, static_cast<int>(nb_levels),
                                                                  in_v, out_v)
      .run();

  // Check correctness of the output
  for(int cell_idx = 0; cell_idx < mesh.cells().size(); ++cell_idx)
    ASSERT_EQ(out_v(cell_idx, 0), 1.0);
}
} // namespace

namespace {
#include <generated_copyEdge.hpp>
TEST(AtlasIntegrationTestCompareOutput, CopyEdge) {
  auto mesh = generateQuadMesh(32, 32);
  size_t nb_levels = 1;

  auto [in_F, in_v] = makeAtlasField("in", mesh.edges().size(), nb_levels);
  auto [out_F, out_v] = makeAtlasField("out", mesh.edges().size(), nb_levels);

  initField(in_v, mesh.edges().size(), nb_levels, 1.0);
  initField(out_v, mesh.edges().size(), nb_levels, -1.0);

  dawn_generated::cxxnaiveico::copyEdge<atlasInterface::atlasTag>(mesh, static_cast<int>(nb_levels),
                                                                  in_v, out_v)
      .run();

  for(int edge_idx = 0; edge_idx < mesh.edges().size(); ++edge_idx)
    ASSERT_EQ(out_v(edge_idx, 0), 1.0);
}
} // namespace

namespace {
#include <generated_verticalSum.hpp>
TEST(AtlasIntegrationTestCompareOutput, verticalCopy) {
  auto mesh = generateQuadMesh(32, 32);
  size_t nb_levels = 5; // must be >= 3

  auto [in_F, in_v] = makeAtlasField("in", mesh.edges().size(), nb_levels);
  auto [out_F, out_v] = makeAtlasField("out", mesh.edges().size(), nb_levels);

  double initValue = 10.;
  initField(in_v, mesh.cells().size(), nb_levels, initValue);
  initField(out_v, mesh.cells().size(), nb_levels, -1.0);

  // Run verticalSum, which just copies the values in the cells above and below into the current
  // level and adds them up
  dawn_generated::cxxnaiveico::verticalSum<atlasInterface::atlasTag>(mesh, nb_levels, in_v, out_v)
      .run();

  // Thats why we expct all the levels except the top and bottom one to hold twice the initial value
  for(int level = 1; level < nb_levels - 1; ++level) {
    for(int cell = 0; cell < mesh.cells().size(); ++cell) {
      ASSERT_EQ(out_v(cell, level), 2 * initValue);
    }
  }
}
} // namespace

namespace {
#include <generated_accumulateEdgeToCell.hpp>
TEST(AtlasIntegrationTestCompareOutput, Accumulate) {
  auto mesh = generateQuadMesh(32, 32);
  size_t nb_levels = 1;

  auto [in_F, in_v] = makeAtlasField("in", mesh.edges().size(), nb_levels);
  auto [out_F, out_v] = makeAtlasField("out", mesh.cells().size(), nb_levels);

  initField(in_v, mesh.edges().size(), nb_levels, 1.0);
  initField(out_v, mesh.cells().size(), nb_levels, -1.0);

  dawn_generated::cxxnaiveico::accumulateEdgeToCell<atlasInterface::atlasTag>(
      mesh, static_cast<int>(nb_levels), in_v, out_v)
      .run();

  // Check correctness of the output
  for(int cell_idx = 0; cell_idx < mesh.cells().size(); ++cell_idx) {
    ASSERT_EQ(out_v(cell_idx, 0), 4.0);
  }
}
} // namespace

namespace {
#include <generated_diffusion.hpp>
#include <reference_diffusion.hpp>
TEST(AtlasIntegrationTestCompareOutput, Diffusion) {
  auto mesh = generateQuadMesh(32, 32);
  size_t nb_levels = 1;

  // Create input (on cells) and output (on cells) fields for generated and reference stencils
  auto [in_ref, in_v_ref] = makeAtlasField("in_v_ref", mesh.cells().size(), nb_levels);
  auto [in_gen, in_v_gen] = makeAtlasField("in_v_gen", mesh.cells().size(), nb_levels);
  auto [out_ref, out_v_ref] = makeAtlasField("out_v_ref", mesh.cells().size(), nb_levels);
  auto [out_gen, out_v_gen] = makeAtlasField("out_v_gen", mesh.cells().size(), nb_levels);

  AtlasToCartesian atlasToCartesianMapper(mesh);

  for(int cellIdx = 0, size = mesh.cells().size(); cellIdx < size; ++cellIdx) {
    auto [cartX, cartY] = atlasToCartesianMapper.cellMidpoint(mesh, cellIdx);
    bool inX = cartX > 0.375 && cartX < 0.625;
    bool inY = cartY > 0.375 && cartY < 0.625;
    in_v_ref(cellIdx, 0) = (inX && inY) ? 1 : 0;
    in_v_gen(cellIdx, 0) = (inX && inY) ? 1 : 0;
  }

  for(int i = 0; i < 5; ++i) {
    // Run the stencils
    dawn_generated::cxxnaiveico::reference_diffusion<atlasInterface::atlasTag>(
        mesh, static_cast<int>(nb_levels), in_v_ref, out_v_ref)
        .run();
    dawn_generated::cxxnaiveico::diffusion<atlasInterface::atlasTag>(
        mesh, static_cast<int>(nb_levels), in_v_gen, out_v_gen)
        .run();

    // Swap in and out
    using std::swap;
    swap(in_ref, out_ref);
    swap(in_gen, out_gen);
  }

  // Check correctness of the output
  {
    auto out_v_ref = atlas::array::make_view<double, 2>(out_ref);
    auto out_v_gen = atlas::array::make_view<double, 2>(out_gen);
    UnstructuredVerifier v;
    EXPECT_TRUE(v.compareArrayView(out_v_gen, out_v_ref)) << "while comparing output (on cells)";
  }
}
} // namespace

namespace {
#include <generated_diamond.hpp>
#include <reference_diamond.hpp>
TEST(AtlasIntegrationTestCompareOutput, Diamond) {
  auto mesh = generateEquilatMesh(32, 32);
  const size_t nb_levels = 1;
  const size_t level = 0;

  // Create input (on cells) and output (on cells) fields for generated and reference stencils
  auto [in_ref, in_v_ref] = makeAtlasField("in_v_ref", mesh.nodes().size(), nb_levels);
  auto [in_gen, in_v_gen] = makeAtlasField("in_v_gen", mesh.nodes().size(), nb_levels);
  auto [out_ref, out_v_ref] = makeAtlasField("out_v_ref", mesh.edges().size(), nb_levels);
  auto [out_gen, out_v_gen] = makeAtlasField("out_v_gen", mesh.edges().size(), nb_levels);

  auto xy = atlas::array::make_view<double, 2>(mesh.nodes().xy());
  for(int nodeIdx = 0, size = mesh.nodes().size(); nodeIdx < size; ++nodeIdx) {
    double x = 0.5 * (xy(nodeIdx, atlas::LON) + xy(nodeIdx, atlas::LON));
    double y = 0.5 * (xy(nodeIdx, atlas::LAT) + xy(nodeIdx, atlas::LAT));
    in_v_ref(nodeIdx, level) = sin(x) * sin(y);
    in_v_gen(nodeIdx, level) = sin(x) * sin(y);
  }

  dawn_generated::cxxnaiveico::reference_diamond<atlasInterface::atlasTag>(
      mesh, static_cast<int>(nb_levels), out_v_ref, in_v_ref)
      .run();
  dawn_generated::cxxnaiveico::diamond<atlasInterface::atlasTag>(mesh, static_cast<int>(nb_levels),
                                                                 out_v_gen, in_v_gen)
      .run();

  // Check correctness of the output
  {
    auto out_v_ref = atlas::array::make_view<double, 2>(out_ref);
    auto out_v_gen = atlas::array::make_view<double, 2>(out_gen);
    UnstructuredVerifier v;
    EXPECT_TRUE(v.compareArrayView(out_v_gen, out_v_ref)) << "while comparing output (on cells)";
  }
}
} // namespace

namespace {
#include <generated_intp.hpp>
#include <reference_intp.hpp>
TEST(AtlasIntegrationTestCompareOutput, Intp) {
  auto mesh = generateEquilatMesh(32, 32);
  const size_t nb_levels = 1;
  const size_t level = 0;

  // Create input (on cells) and output (on cells) fields for generated and reference stencils
  auto [in_ref, in_v_ref] = makeAtlasField("in_v_ref", mesh.cells().size(), nb_levels);
  auto [in_gen, in_v_gen] = makeAtlasField("in_v_gen", mesh.cells().size(), nb_levels);
  auto [out_ref, out_v_ref] = makeAtlasField("out_v_ref", mesh.cells().size(), nb_levels);
  auto [out_gen, out_v_gen] = makeAtlasField("out_v_gen", mesh.cells().size(), nb_levels);

  auto xy = atlas::array::make_view<double, 2>(mesh.nodes().xy());
  for(int cellIdx = 0, size = mesh.cells().size(); cellIdx < size; ++cellIdx) {
    int v0 = mesh.cells().node_connectivity()(cellIdx, 0);
    int v1 = mesh.cells().node_connectivity()(cellIdx, 1);
    int v2 = mesh.cells().node_connectivity()(cellIdx, 2);
    double x = 1. / 3. * (xy(v0, atlas::LON) + xy(v1, atlas::LON) + xy(v2, atlas::LON));
    double y = 1. / 3. * (xy(v0, atlas::LAT) + xy(v1, atlas::LAT) + xy(v2, atlas::LON));
    in_v_ref(cellIdx, level) = sin(x) * sin(y);
    in_v_gen(cellIdx, level) = sin(x) * sin(y);
  }

  dawn_generated::cxxnaiveico::reference_intp<atlasInterface::atlasTag>(
      mesh, static_cast<int>(nb_levels), in_v_ref, out_v_ref)
      .run();
  dawn_generated::cxxnaiveico::intp<atlasInterface::atlasTag>(mesh, static_cast<int>(nb_levels),
                                                              in_v_gen, out_v_gen)
      .run();

  // Check correctness of the output
  {
    auto out_v_ref = atlas::array::make_view<double, 2>(out_ref);
    auto out_v_gen = atlas::array::make_view<double, 2>(out_gen);
    UnstructuredVerifier v;
    EXPECT_TRUE(v.compareArrayView(out_v_gen, out_v_ref)) << "while comparing output (on cells)";
  }
}
} // namespace

namespace {
#include <generated_gradient.hpp>
#include <reference_gradient.hpp>

void build_periodic_edges(atlas::Mesh& mesh, int nx, int ny, const AtlasToCartesian& atlasMapper) {
  atlas::mesh::HybridElements::Connectivity& edgeCellConnectivity =
      mesh.edges().cell_connectivity();
  const int missingVal = edgeCellConnectivity.missing_value();

  auto unhash = [](int idx, int nx) -> std::tuple<int, int> {
    int j = idx / nx;
    int i = idx - (j * nx);
    return {i, j};
  };

  for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
    int numNbh = edgeCellConnectivity.cols(edgeIdx);
    assert(numNbh == 2);

    int nbhLo = edgeCellConnectivity(edgeIdx, 0);
    int nbhHi = edgeCellConnectivity(edgeIdx, 1);

    assert(!(nbhLo == missingVal && nbhHi == missingVal));

    // if we encountered a missing value, we need to fix the neighbor list
    if(nbhLo == missingVal || nbhHi == missingVal) {
      int validIdx = (nbhLo == missingVal) ? nbhHi : nbhLo;
      auto [cellI, cellJ] = unhash(validIdx, nx);
      // depending whether we are vertical or horizontal, we need to reflect either the first or
      // second index
      if(atlasMapper.edgeOrientation(mesh, edgeIdx) == Orientation::Vertical) {
        assert(cellI == nx - 1 || cellI == 0);
        cellI = (cellI == nx - 1) ? 0 : nx - 1;
      } else { // Orientation::Horizontal
        assert(cellJ == ny - 1 || cellJ == 0);
        cellJ = (cellJ == ny - 1) ? 0 : ny - 1;
      }
      int oppositeIdx = cellI + cellJ * nx;
      // ammend the neighbor list
      if(nbhLo == missingVal) {
        edgeCellConnectivity.set(edgeIdx, 0, oppositeIdx);
      } else {
        edgeCellConnectivity.set(edgeIdx, 1, oppositeIdx);
      }
    }
  }
}

TEST(AtlasIntegrationTestCompareOutput, Gradient) {
  // this test computes a gradient in a periodic domain
  //
  //   this is  achieved by reducing a signal from a cell
  //   field onto the edges using the weights [1, -1]. This is equiavlent
  //   to a second order finite difference stencils, missing the division by the cell spacing
  //   (currently omitted).
  //
  //   after this first step, vertical edges contain the x gradient and horizontal edges contain the
  //   y gradient of the original signal. to get the x gradients on the cells (in order to properly
  //   visualize them) the edges are reduced again onto the cells, using weights [0.5, 0, 0, 0.5]
  //
  //   this test uses the AtlasCartesianMapper to assign values

  // kept low for now to get easy debug-able output
  const int numCell = 10;

  // apparently, one needs to be added to the second dimension in order to get a
  // square mesh, or we are mis-interpreting the output
  auto mesh = generateQuadMesh(numCell, numCell + 1);

  AtlasToCartesian atlasToCartesianMapper(mesh);
  build_periodic_edges(mesh, numCell, numCell, atlasToCartesianMapper);

  int nb_levels = 1;

  auto [ref_cells, ref_cells_v] = makeAtlasField("ref_cells", mesh.cells().size(), nb_levels);
  auto [ref_edges, ref_edges_v] = makeAtlasField("ref_edges", mesh.edges().size(), nb_levels);
  auto [gen_cells, gen_cells_v] = makeAtlasField("gen_cells", mesh.cells().size(), nb_levels);
  auto [gen_edges, gen_edges_v] = makeAtlasField("gen_edges", mesh.edges().size(), nb_levels);

  for(int cellIdx = 0, size = mesh.cells().size(); cellIdx < size; ++cellIdx) {
    auto [cartX, cartY] = atlasToCartesianMapper.cellMidpoint(mesh, cellIdx);
    double val =
        sin(cartX * M_PI) * sin(cartY * M_PI); // periodic signal fitting periodic boundaries
    ref_cells_v(cellIdx, 0) = val;
    gen_cells_v(cellIdx, 0) = val;
  }

  dawn_generated::cxxnaiveico::reference_gradient<atlasInterface::atlasTag>(
      mesh, nb_levels, ref_cells_v, ref_edges_v)
      .run();
  dawn_generated::cxxnaiveico::gradient<atlasInterface::atlasTag>(mesh, nb_levels, gen_cells_v,
                                                                  gen_edges_v)
      .run();

  // Check correctness of the output
  {
    auto ref_cells_v = atlas::array::make_view<double, 2>(ref_cells);
    auto gen_cells_v = atlas::array::make_view<double, 2>(gen_cells);
    UnstructuredVerifier v;
    EXPECT_TRUE(v.compareArrayView(ref_cells_v, gen_cells_v))
        << "while comparing output (on cells)";
  }
}
} // namespace

namespace {
#include <generated_verticalSolver.hpp>
TEST(AtlasIntegrationTestCompareOutput, verticalSolver) {
  const int numCell = 5;

  // This tests the unstructured vertical solver
  // A small system with a manufactured solution is generated for each cell

  // apparently, one needs to be added to the second dimension in order to get a
  // square mesh, or we are mis-interpreting the output
  auto mesh = generateQuadMesh(numCell, numCell + 1);

  // the 4 fields required for the thomas algorithm
  //  c.f.
  //  https://en.wikibooks.org/wiki/Algorithm_Implementation/Linear_Algebra/Tridiagonal_matrix_algorithm#C
  int nb_levels = 5;
  auto [a_f, a_v] = makeAtlasField("a", mesh.cells().size(), nb_levels);
  auto [b_f, b_v] = makeAtlasField("b", mesh.cells().size(), nb_levels);
  auto [c_f, c_v] = makeAtlasField("c", mesh.cells().size(), nb_levels);
  auto [d_f, d_v] = makeAtlasField("d", mesh.cells().size(), nb_levels);

  // solution to this problem will be [1,2,3,4,5] at each cell location
  for(int cell = 0; cell < mesh.cells().size(); ++cell) {
    for(int k = 0; k < nb_levels; k++) {
      a_v(cell, k) = k + 1;
      b_v(cell, k) = k + 1;
      c_v(cell, k) = k + 2;
    }

    d_v(cell, 0) = 5;
    d_v(cell, 1) = 15;
    d_v(cell, 2) = 31;
    d_v(cell, 3) = 53;
    d_v(cell, 4) = 45;
  }

  dawn_generated::cxxnaiveico::tridiagonalSolve<atlasInterface::atlasTag>(mesh, nb_levels, a_v, b_v,
                                                                          c_v, d_v)
      .run();

  for(int cell = 0; cell < mesh.cells().size(); ++cell) {
    for(int k = 0; k < nb_levels; k++) {
      EXPECT_TRUE(abs(d_v(cell, k) - (k + 1)) < 1e3 * std::numeric_limits<double>::epsilon());
    }
  }
}

namespace {
#include <generated_NestedSimple.hpp>
TEST(AtlasIntegrationTestCompareOutput, nestedSimple) {
  const int numCell = 10;
  auto mesh = generateQuadMesh(numCell, numCell + 1);

  int nb_levels = 1;
  auto [cells, v_cells] = makeAtlasField("cells", mesh.cells().size(), nb_levels);
  auto [edges, v_edges] = makeAtlasField("edges", mesh.edges().size(), nb_levels);
  auto [nodes, v_nodes] = makeAtlasField("nodes", mesh.nodes().size(), nb_levels);

  initField(v_nodes, mesh.nodes().size(), nb_levels, 1.);

  dawn_generated::cxxnaiveico::nestedSimple<atlasInterface::atlasTag>(mesh, nb_levels, v_cells,
                                                                      v_edges, v_nodes)
      .run();

  // each vertex stores value 1                 1
  // vertices are reduced onto edges            2
  // each face reduces its edges (4 per face)   8
  for(int i = 0; i < mesh.cells().size(); i++) {
    EXPECT_TRUE(fabs(v_cells(i, 0) - 8) < 1e3 * std::numeric_limits<double>::epsilon());
  }
}
} // namespace

namespace {
#include <generated_NestedWithField.hpp>
TEST(AtlasIntegrationTestCompareOutput, nestedWithField) {
  const int numCell = 10;
  auto mesh = generateQuadMesh(numCell, numCell + 1);

  int nb_levels = 1;
  auto [cells, v_cells] = makeAtlasField("cells", mesh.cells().size(), nb_levels);
  auto [edges, v_edges] = makeAtlasField("edges", mesh.edges().size(), nb_levels);
  auto [nodes, v_nodes] = makeAtlasField("nodes", mesh.nodes().size(), nb_levels);

  initField(v_nodes, mesh.nodes().size(), nb_levels, 1.);
  initField(v_edges, mesh.edges().size(), nb_levels, 200.);

  dawn_generated::cxxnaiveico::nestedWithField<atlasInterface::atlasTag>(mesh, nb_levels, v_cells,
                                                                         v_edges, v_nodes)
      .run();

  // each vertex stores value 1                 1
  // vertices are reduced onto edges            2
  // each edge stores 200                     202
  // each face reduces its edges (4 per face) 808
  for(int i = 0; i < mesh.cells().size(); i++) {
    EXPECT_TRUE(fabs(v_cells(i, 0) - 808) < 1e3 * std::numeric_limits<double>::epsilon());
  }
}
} // namespace

namespace {
#include <generated_sparseDimension.hpp>
TEST(AtlasIntegrationTestCompareOutput, sparseDimensions) {
  auto mesh = generateQuadMesh(10, 11);
  const int edgesPerCell = 4;
  const int nb_levels = 1;

  auto [cells_F, cells_v] = makeAtlasField("cells", mesh.cells().size(), nb_levels);
  auto [edges_F, edges_v] = makeAtlasField("edges", mesh.edges().size(), nb_levels);
  auto [sparseDim_F, sparseDim_v] =
      makeAtlasSparseField("sparse", mesh.cells().size(), edgesPerCell, nb_levels);

  initSparseField(sparseDim_v, mesh.cells().size(), nb_levels, edgesPerCell, 200.);
  initField(edges_v, mesh.edges().size(), nb_levels, 1.);

  dawn_generated::cxxnaiveico::sparseDimension<atlasInterface::atlasTag>(mesh, nb_levels, cells_v,
                                                                         edges_v, sparseDim_v)
      .run();

  // each edge stores 1                                         1
  // this is multiplied by the sparse dim storing 200         200
  // this is reduced by sum onto the cells at 4 eges p cell   800
  for(int i = 0; i < mesh.cells().size(); i++) {
    EXPECT_TRUE(fabs(cells_v(i, 0) - 800) < 1e3 * std::numeric_limits<double>::epsilon());
  }
}
} // namespace

namespace {
#include <generated_sparseDimensionTwice.hpp>
TEST(AtlasIntegrationTestCompareOutput, sparseDimensionsTwice) {
  auto mesh = generateQuadMesh(10, 11);
  const int edgesPerCell = 4;
  const int nb_levels = 1;

  auto [cells_F, cells_v] = makeAtlasField("cells", mesh.cells().size(), nb_levels);
  auto [edges_F, edges_v] = makeAtlasField("edges", mesh.edges().size(), nb_levels);
  auto [sparseDim_F, sparseDim_v] =
      makeAtlasSparseField("sparse", mesh.cells().size(), edgesPerCell, nb_levels);

  initSparseField(sparseDim_v, mesh.cells().size(), nb_levels, edgesPerCell, 200.);
  initField(edges_v, mesh.edges().size(), nb_levels, 1.);

  dawn_generated::cxxnaiveico::sparseDimensionTwice<atlasInterface::atlasTag>(
      mesh, nb_levels, cells_v, edges_v, sparseDim_v)
      .run();

  // each edge stores 1                                                1
  // this is multiplied by the sparse dim storing 200                200
  // this is reduced by sum onto the cells at 4 eges p cell          800
  for(int i = 0; i < mesh.cells().size(); i++) {
    EXPECT_TRUE(fabs(cells_v(i, 0) - 800) < 1e3 * std::numeric_limits<double>::epsilon());
  }
  // NOTE that the second reduction simply overwrites the result of the first one since there is
  // "+=" in the IIRBuilder currently
}
} // namespace
} // namespace
} // namespace