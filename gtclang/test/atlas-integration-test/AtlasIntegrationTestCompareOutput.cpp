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
#include <gridtools/clang_dsl.hpp>

#include "AtlasVerifyer.h"
#include "atlas/functionspace/CellColumns.h"
#include "atlas/functionspace/EdgeColumns.h"
#include "atlas/grid.h"
#include "atlas/mesh/actions/BuildCellCentres.h"
#include "atlas/mesh/actions/BuildEdges.h"
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

  // We only need one vertical level
  size_t nb_levels = 1;

  // Create input (on edges) and output (on edges) fields
  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field in{fs_edges.createField<double>(atlas::option::name("in"))};
  atlas::Field out{fs_edges.createField<double>(atlas::option::name("out"))};

  // Add edges to mesh.edges()
  atlas::mesh::actions::build_edges(mesh);

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

  // We only need one vertical level
  size_t nb_levels = 1;

  // Create input (on edges) and output (on cells) fields
  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field in{fs_edges.createField<double>(atlas::option::name("in"))};

  atlas::functionspace::CellColumns fs_cells(mesh, atlas::option::levels(nb_levels));
  atlas::Field out{fs_cells.createField<double>(atlas::option::name("out"))};

  // Add edges to mesh.edges()
  atlas::mesh::actions::build_edges(mesh);
  // Build connectivity matrix for cells-edges
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);

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
  for(int cell_idx = 0; cell_idx < mesh.cells().size(); ++cell_idx)
    ASSERT_EQ(out_v(cell_idx, 0), 4.0);
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

  // We only need one vertical level
  size_t nb_levels = 1;

  // Create input (on cells) and output (on cells) fields for generated and reference stencils
  atlas::functionspace::CellColumns fs(mesh, atlas::option::levels(nb_levels));
  atlas::Field out_ref{fs.createField<double>(atlas::option::name("out"))};
  atlas::Field out_gen{fs.createField<double>(atlas::option::name("out"))};

  atlas::Field in_ref{fs.createField<double>(atlas::option::name("in"))};
  atlas::Field in_gen{fs.createField<double>(atlas::option::name("in"))};

  // Add edges to mesh.edges()
  atlas::mesh::actions::build_edges(mesh);
  // Build connectivity matrix for cells-edges
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);

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

  for(int i = 0; i < 500; ++i) {

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
    AtlasVerifyer v;
    EXPECT_TRUE(v.compareArrayView(out_v_gen, out_v_ref)) << "while comparing output (on cells)";
  }
}
} // namespace