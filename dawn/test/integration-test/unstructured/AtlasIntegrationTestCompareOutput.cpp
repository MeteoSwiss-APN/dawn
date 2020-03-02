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
#include "AtlasVerifier.h"
#include "atlas/functionspace/CellColumns.h"
#include "atlas/functionspace/EdgeColumns.h"
#include "atlas/grid.h"
#include "atlas/mesh/actions/BuildCellCentres.h"
#include "atlas/mesh/actions/BuildEdges.h"
#include "atlas/mesh/actions/BuildPeriodicBoundaries.h"
#include "atlas/meshgenerator.h"
#include "atlas/option/Options.h"
#include "atlas/output/Gmsh.h"
#include "interface/atlas_interface.hpp"

#include <gtest/gtest.h>

#include <generated_copyCell.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, CopyCell) {
  // Setup a 32 by 32 grid of quads and generate a mesh out of it
  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  // We only need one vertical level
  size_t nb_levels = 1;

  // Create input (on cells) and output (on cells) fields
  atlas::functionspace::CellColumns fs_cells(mesh, atlas::option::levels(nb_levels));
  atlas::Field in{fs_cells.createField<double>(atlas::option::name("in"))};
  atlas::Field out{fs_cells.createField<double>(atlas::option::name("out"))};

  // Make views on the fields (needed to access the field like an array)
  atlasInterface::Field<double> in_v = atlas::array::make_view<double, 2>(in);
  atlasInterface::Field<double> out_v = atlas::array::make_view<double, 2>(out);

  // Initialize fields with data
  for(int cell_idx = 0; cell_idx < mesh.cells().size(); ++cell_idx) {
    in_v(cell_idx, 0) = 1.0;
    out_v(cell_idx, 0) = -1.0;
  }

  // Run the stencil
  dawn_generated::cxxnaiveico::copyCell<atlasInterface::atlasTag>(mesh, static_cast<int>(nb_levels),
                                                                  in_v, out_v)
      .run();

  // Check correctness of the output
  for(int cell_idx = 0; cell_idx < mesh.cells().size(); ++cell_idx)
    ASSERT_EQ(out_v(cell_idx, 0), 1.0);
}
} // namespace

#include <generated_copyEdge.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, CopyEdge) {
  // Setup a 32 by 32 grid of quads and generate a mesh out of it
  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);
  // Add edges to mesh.edges()
  atlas::mesh::actions::build_edges(mesh);

  // We only need one vertical level
  size_t nb_levels = 1;

  // Create input (on edges) and output (on edges) fields
  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field in{fs_edges.createField<double>(atlas::option::name("in"))};
  atlas::Field out{fs_edges.createField<double>(atlas::option::name("out"))};

  // Make views on the fields (needed to access the field like an array)
  atlasInterface::Field<double> in_v = atlas::array::make_view<double, 2>(in);
  atlasInterface::Field<double> out_v = atlas::array::make_view<double, 2>(out);

  // Initialize fields with data
  for(int edge_idx = 0; edge_idx < mesh.edges().size(); ++edge_idx) {
    in_v(edge_idx, 0) = 1.0;
    out_v(edge_idx, 0) = -1.0;
  }

  // Run the stencil
  dawn_generated::cxxnaiveico::copyEdge<atlasInterface::atlasTag>(mesh, static_cast<int>(nb_levels),
                                                                  in_v, out_v)
      .run();

  // Check correctness of the output
  for(int edge_idx = 0; edge_idx < mesh.edges().size(); ++edge_idx)
    ASSERT_EQ(out_v(edge_idx, 0), 1.0);
}
} // namespace

#include <generated_verticalSum.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, verticalCopy) {
  // Setup a 32 by 32 grid of quads and generate a mesh out of it
  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  // We use 5 levels, but any number >= 3 is fine
  size_t nb_levels = 1;

  // We construct an in and out Field on cells
  atlas::functionspace::CellColumns fs(mesh, atlas::option::levels(nb_levels));
  atlas::Field out{fs.createField<double>(atlas::option::name("out"))};
  atlas::Field in{fs.createField<double>(atlas::option::name("in"))};

  // Make views on the fields (needed to access the field like an array)
  atlasInterface::Field<double> in_v = atlas::array::make_view<double, 2>(in);
  atlasInterface::Field<double> out_v = atlas::array::make_view<double, 2>(out);

  // Initialize input
  double init_value = 10;
  for(int level = 0; level < nb_levels; ++level) {
    for(int cell = 0; cell < mesh.cells().size(); ++cell) {
      in_v(cell, level) = init_value;
    }
  }

  // Run verticalSum, which just copies the values in the cells above and below into the current
  // level and adds them up
  dawn_generated::cxxnaiveico::verticalSum<atlasInterface::atlasTag>(mesh, nb_levels, in_v, out_v)
      .run();

  // Thats why we expct all the levels except the top and bottom one to hold twice the initial value
  for(int level = 1; level < nb_levels - 1; ++level) {
    for(int cell = 0; cell < mesh.cells().size(); ++cell) {
      ASSERT_EQ(out_v(cell, level), 2 * init_value);
    }
  }
}
} // namespace

#include <generated_accumulateEdgeToCell.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, Accumulate) {
  // Setup a 32 by 32 grid of quads and generate a mesh out of it
  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  // Add edges to mesh.edges()
  atlas::mesh::actions::build_edges(mesh);
  // Build connectivity matrix for cells-edges
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);

  // We only need one vertical level
  size_t nb_levels = 1;

  // Create input (on edges) and output (on cells) fields
  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field in{fs_edges.createField<double>(atlas::option::name("in"))};

  atlas::functionspace::CellColumns fs_cells(mesh, atlas::option::levels(nb_levels));
  atlas::Field out{fs_cells.createField<double>(atlas::option::name("out"))};

  // Make views on the fields (needed to access the field like an array)
  atlasInterface::Field<double> in_v = atlas::array::make_view<double, 2>(in);
  atlasInterface::Field<double> out_v = atlas::array::make_view<double, 2>(out);

  // Initialize fields with data
  for(int edge_idx = 0; edge_idx < mesh.edges().size(); ++edge_idx)
    in_v(edge_idx, 0) = 1.0;

  for(int cell_idx = 0; cell_idx < mesh.cells().size(); ++cell_idx)
    out_v(cell_idx, 0) = -1.0;

  // Run the stencil
  dawn_generated::cxxnaiveico::accumulateEdgeToCell<atlasInterface::atlasTag>(
      mesh, static_cast<int>(nb_levels), in_v, out_v)
      .run();

  // Check correctness of the output
  for(int cell_idx = 0; cell_idx < mesh.cells().size(); ++cell_idx) {
    ASSERT_EQ(out_v(cell_idx, 0), 4.0);
  }
}
} // namespace

#include <generated_diffusion.hpp>
#include <reference_diffusion.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, Diffusion) {
  // Setup a 32 by 32 grid of quads and generate a mesh out of it
  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);
  // Add edges to mesh.edges()
  atlas::mesh::actions::build_edges(mesh);
  // Build connectivity matrix for cells-edges
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);

  // We only need one vertical level
  size_t nb_levels = 1;

  // Create input (on cells) and output (on cells) fields for generated and reference stencils
  atlas::functionspace::CellColumns fs(mesh, atlas::option::levels(nb_levels));
  atlas::Field out_ref{fs.createField<double>(atlas::option::name("out"))};
  atlas::Field out_gen{fs.createField<double>(atlas::option::name("out"))};

  atlas::Field in_ref{fs.createField<double>(atlas::option::name("in"))};
  atlas::Field in_gen{fs.createField<double>(atlas::option::name("in"))};

  // Initialize fields with data
  {
    // Make views on the fields (needed to access the field like an array)
    auto in_v_ref = atlas::array::make_view<double, 2>(in_ref);
    auto in_v_gen = atlas::array::make_view<double, 2>(in_gen);
    auto out_v_ref = atlas::array::make_view<double, 2>(out_ref);
    auto out_v_gen = atlas::array::make_view<double, 2>(out_gen);

    auto const& node_connectivity = mesh.cells().node_connectivity();
    const double rpi = 2.0 * asin(1.0);
    const double deg2rad = rpi / 180.;
    auto lonlat = atlas::array::make_view<double, 2>(mesh.nodes().lonlat());

    auto lon0 = lonlat(node_connectivity(0, 0), 0);
    auto lat0 = lonlat(node_connectivity(0, 0), 1);
    auto lon1 = lonlat(node_connectivity(mesh.cells().size() - 1, 0), 0);
    auto lat1 = lonlat(node_connectivity(mesh.cells().size() - 1, 0), 1);
    for(int jCell = 0, size = mesh.cells().size(); jCell < size; ++jCell) {
      double center_x =
          (lonlat(node_connectivity(jCell, 0), 0) - lon0 + (lon0 - lon1) / 2.f) * deg2rad;
      double center_y =
          (lonlat(node_connectivity(jCell, 0), 1) - lat0 + (lat0 - lat1) / 2.f) * deg2rad;
      in_v_ref(jCell, 0) = std::abs(center_x) < .5 && std::abs(center_y) < .5 ? 1 : 0;
      in_v_gen(jCell, 0) = std::abs(center_x) < .5 && std::abs(center_y) < .5 ? 1 : 0;
      out_v_ref(jCell, 0) = -1.0;
      out_v_gen(jCell, 0) = -1.0;
    }
  }

  for(int i = 0; i < 5; ++i) {

    // Make views on the fields (needed to access the field like an array)
    atlasInterface::Field<double> in_v_ref = atlas::array::make_view<double, 2>(in_ref);
    atlasInterface::Field<double> in_v_gen = atlas::array::make_view<double, 2>(in_gen);
    atlasInterface::Field<double> out_v_ref = atlas::array::make_view<double, 2>(out_ref);
    atlasInterface::Field<double> out_v_gen = atlas::array::make_view<double, 2>(out_gen);

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
    AtlasVerifier v;
    EXPECT_TRUE(v.compareArrayView(out_v_gen, out_v_ref)) << "while comparing output (on cells)";
  }
}
} // namespace

#include <generated_gradient.hpp>
#include <reference_gradient.hpp>
namespace {

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
  atlas::StructuredGrid structuredGrid =
      atlas::Grid("L" + std::to_string(numCell) + "x" + std::to_string(numCell + 1));
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);
  atlas::mesh::actions::build_edges(
      mesh, atlas::util::Config("pole_edges", false)); // work around to eliminate pole edges
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
  atlas::mesh::actions::build_element_to_edge_connectivity(mesh);

  AtlasToCartesian atlasToCartesianMapper(mesh);
  build_periodic_edges(mesh, numCell, numCell, atlasToCartesianMapper);

  int nb_levels = 1;
  atlas::functionspace::CellColumns fs_cols(mesh, atlas::option::levels(nb_levels));
  atlas::Field ref_cells_f{fs_cols.createField<double>(atlas::option::name("ref_cells"))};
  atlas::Field gen_cells_f{fs_cols.createField<double>(atlas::option::name("gen_cells"))};

  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field ref_edges_f{fs_edges.createField<double>(atlas::option::name("ref_edges"))};
  atlas::Field gen_edges_f{fs_edges.createField<double>(atlas::option::name("gen_edges"))};

  atlasInterface::Field<double> ref_cells_v = atlas::array::make_view<double, 2>(ref_cells_f);
  atlasInterface::Field<double> gen_cells_v = atlas::array::make_view<double, 2>(gen_cells_f);
  atlasInterface::Field<double> ref_edges_v = atlas::array::make_view<double, 2>(ref_edges_f);
  atlasInterface::Field<double> gen_edges_v = atlas::array::make_view<double, 2>(gen_edges_f);

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
    auto ref_cells_v = atlas::array::make_view<double, 2>(ref_cells_f);
    auto gen_cells_v = atlas::array::make_view<double, 2>(gen_cells_f);
    AtlasVerifier v;
    EXPECT_TRUE(v.compareArrayView(ref_cells_v, gen_cells_v))
        << "while comparing output (on cells)";
  }
}
} // namespace

#include <generated_verticalSolver.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, verticalSolver) {
  const int numCell = 5;

  // This tests the unstructured vertical solver
  // A small system with a manufactured solution is generated for each cell

  // apparently, one needs to be added to the second dimension in order to get a
  // square mesh, or we are mis-interpreting the output
  atlas::StructuredGrid structuredGrid =
      atlas::Grid("L" + std::to_string(numCell) + "x" + std::to_string(numCell + 1));
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  // the 4 fields required for the thomas algorithm
  //  c.f.
  //  https://en.wikibooks.org/wiki/Algorithm_Implementation/Linear_Algebra/Tridiagonal_matrix_algorithm#C
  int nb_levels = 5;
  atlas::functionspace::CellColumns fs_cols(mesh, atlas::option::levels(nb_levels));
  atlas::Field a_f{fs_cols.createField<double>(atlas::option::name("a_cells"))};
  atlas::Field b_f{fs_cols.createField<double>(atlas::option::name("b_cells"))};
  atlas::Field c_f{fs_cols.createField<double>(atlas::option::name("c_cells"))};
  atlas::Field d_f{fs_cols.createField<double>(atlas::option::name("d_cells"))};

  atlasInterface::Field<double> a_v = atlas::array::make_view<double, 2>(a_f);
  atlasInterface::Field<double> b_v = atlas::array::make_view<double, 2>(b_f);
  atlasInterface::Field<double> c_v = atlas::array::make_view<double, 2>(c_f);
  atlasInterface::Field<double> d_v = atlas::array::make_view<double, 2>(d_f);

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

#include <generated_NestedSimple.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, nestedSimple) {
  const int numCell = 10;

  // apparently, one needs to be added to the second dimension in order to get a
  // square mesh, or we are mis-interpreting the output
  atlas::StructuredGrid structuredGrid =
      atlas::Grid("L" + std::to_string(numCell) + "x" + std::to_string(numCell + 1));
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  atlas::mesh::actions::build_edges(mesh, atlas::util::Config("pole_edges", false));
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
  atlas::mesh::actions::build_element_to_edge_connectivity(mesh);

  int nb_levels = 1;
  atlas::functionspace::CellColumns fs_cells(mesh, atlas::option::levels(nb_levels));
  atlas::Field f_cells{fs_cells.createField<double>(atlas::option::name("cells"))};

  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field f_edges{fs_edges.createField<double>(atlas::option::name("edges"))};

  atlas::functionspace::EdgeColumns fs_vertices(mesh, atlas::option::levels(nb_levels));
  atlas::Field f_vertices{fs_vertices.createField<double>(atlas::option::name("vertices"))};

  atlasInterface::Field<double> v_cells = atlas::array::make_view<double, 2>(f_cells);
  atlasInterface::Field<double> v_edges = atlas::array::make_view<double, 2>(f_edges);
  atlasInterface::Field<double> v_vertices = atlas::array::make_view<double, 2>(f_vertices);

  for(int i = 0; i < mesh.nodes().size(); i++) {
    v_vertices(i, 0) = 1;
  }

  dawn_generated::cxxnaiveico::nestedSimple<atlasInterface::atlasTag>(mesh, nb_levels, v_cells,
                                                                      v_edges, v_vertices)
      .run();

  // each vertex stores value 1                 1
  // vertices are reduced onto edges            2
  // each face reduces its edges (4 per face)   8
  for(int i = 0; i < mesh.cells().size(); i++) {
    EXPECT_TRUE(fabs(v_cells(i, 0) - 8) < 1e3 * std::numeric_limits<double>::epsilon());
  }
}
} // namespace

#include <generated_NestedWithField.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, nestedWithField) {
  const int numCell = 10;

  // apparently, one needs to be added to the second dimension in order to get a
  // square mesh, or we are mis-interpreting the output
  atlas::StructuredGrid structuredGrid =
      atlas::Grid("L" + std::to_string(numCell) + "x" + std::to_string(numCell + 1));
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  atlas::mesh::actions::build_edges(mesh, atlas::util::Config("pole_edges", false));
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
  atlas::mesh::actions::build_element_to_edge_connectivity(mesh);

  int nb_levels = 1;
  atlas::functionspace::CellColumns fs_cells(mesh, atlas::option::levels(nb_levels));
  atlas::Field f_cells{fs_cells.createField<double>(atlas::option::name("cells"))};

  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field f_edges{fs_edges.createField<double>(atlas::option::name("edges"))};

  atlas::functionspace::EdgeColumns fs_vertices(mesh, atlas::option::levels(nb_levels));
  atlas::Field f_vertices{fs_vertices.createField<double>(atlas::option::name("edges"))};

  atlasInterface::Field<double> v_cells = atlas::array::make_view<double, 2>(f_cells);
  atlasInterface::Field<double> v_edges = atlas::array::make_view<double, 2>(f_edges);
  atlasInterface::Field<double> v_vertices = atlas::array::make_view<double, 2>(f_vertices);

  for(int i = 0; i < mesh.nodes().size(); i++) {
    v_vertices(i, 0) = 1;
  }

  for(int i = 0; i < mesh.edges().size(); i++) {
    v_edges(i, 0) = 200;
  }

  dawn_generated::cxxnaiveico::nestedWithField<atlasInterface::atlasTag>(mesh, nb_levels, v_cells,
                                                                         v_edges, v_vertices)
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

#include <generated_sparseDimension.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, sparseDimensions) {
  atlas::StructuredGrid structuredGrid = atlas::Grid("L10x11");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);
  atlas::mesh::actions::build_edges(mesh, atlas::util::Config("pole_edges", false));
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
  atlas::mesh::actions::build_element_to_edge_connectivity(mesh);

  const int edgesPerCell = 4;

  int nb_levels = 1;
  atlas::functionspace::CellColumns fs_cells(mesh, atlas::option::levels(nb_levels));
  atlas::Field cellsField{fs_cells.createField<double>(atlas::option::name("cells"))};
  atlas::Field sparseDimension{fs_cells.createField<double>(
      atlas::option::name("sparseDimension") | atlas::option::variables(edgesPerCell))};

  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field edgesField{fs_edges.createField<double>(atlas::option::name("edges"))};

  atlasInterface::Field<double> cells_v = atlas::array::make_view<double, 2>(cellsField);
  atlasInterface::Field<double> edges_v = atlas::array::make_view<double, 2>(edgesField);
  atlasInterface::SparseDimension<double> sparseDim_v =
      atlas::array::make_view<double, 3>(sparseDimension);

  const int level = 0;
  for(int iCell = 0; iCell < mesh.cells().size(); iCell++) {
    cells_v(iCell, level) = 0;
    for(int jNbh = 0; jNbh < edgesPerCell; jNbh++) {
      sparseDim_v(iCell, jNbh, level) = 200;
    }
  }

  for(int iEdge = 0; iEdge < mesh.edges().size(); iEdge++) {
    edges_v(iEdge, level) = 1;
  }

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

#include <generated_sparseDimensionTwice.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, sparseDimensionsTwice) {
  // The purpose of this test is to ensure that the sparse index is handled correctly
  // across multiple reductions
  atlas::StructuredGrid structuredGrid = atlas::Grid("L10x11");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);
  atlas::mesh::actions::build_edges(mesh, atlas::util::Config("pole_edges", false));
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
  atlas::mesh::actions::build_element_to_edge_connectivity(mesh);

  const int edgesPerCell = 4;

  int nb_levels = 1;
  atlas::functionspace::CellColumns fs_cells(mesh, atlas::option::levels(nb_levels));
  atlas::Field cellsField{fs_cells.createField<double>(atlas::option::name("cells"))};
  atlas::Field sparseDimension{fs_cells.createField<double>(
      atlas::option::name("sparseDimension") | atlas::option::variables(edgesPerCell))};

  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field edgesField{fs_edges.createField<double>(atlas::option::name("edges"))};

  atlasInterface::Field<double> cells_v = atlas::array::make_view<double, 2>(cellsField);
  atlasInterface::Field<double> edges_v = atlas::array::make_view<double, 2>(edgesField);
  atlasInterface::SparseDimension<double> sparseDim_v =
      atlas::array::make_view<double, 3>(sparseDimension);

  const int level = 0;
  for(int iCell = 0; iCell < mesh.cells().size(); iCell++) {
    cells_v(iCell, level) = 0;
    for(int jNbh = 0; jNbh < edgesPerCell; jNbh++) {
      sparseDim_v(iCell, jNbh, level) = 200;
    }
  }

  for(int iEdge = 0; iEdge < mesh.edges().size(); iEdge++) {
    edges_v(iEdge, level) = 1;
  }

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
