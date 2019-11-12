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
#include <fstream>
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
  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  size_t nb_levels = 1;

  atlas::functionspace::CellColumns fs_cells(mesh, atlas::option::levels(nb_levels));
  atlas::Field in{fs_cells.createField<double>(atlas::option::name("in"))};
  atlas::Field out{fs_cells.createField<double>(atlas::option::name("out"))};

  atlasInterface::Field<double> in_v = atlas::array::make_view<double, 2>(in);
  atlasInterface::Field<double> out_v = atlas::array::make_view<double, 2>(out);

  for(int cell_idx = 0; cell_idx < mesh.cells().size(); ++cell_idx)
    in_v(cell_idx, 0) = 1.0;

  dawn_generated::cxxnaiveico::copyCell<atlasInterface::atlasTag>(mesh, static_cast<int>(nb_levels),
                                                                  in_v, out_v)
      .run();

  for(int cell_idx = 0; cell_idx < mesh.cells().size(); ++cell_idx)
    ASSERT_EQ(out_v(cell_idx, 0), 1.0);
}
} // namespace

#include <generated_copyEdge.hpp>
namespace {
// TODO: this is currently broken, because cannot construct with IIRBuilder a stage with location
// type different from cells
TEST(AtlasIntegrationTestCompareOutput, CopyEdge) {
  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  size_t nb_levels = 1;

  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field in{fs_edges.createField<double>(atlas::option::name("in"))};
  atlas::Field out{fs_edges.createField<double>(atlas::option::name("out"))};

  atlas::mesh::actions::build_edges(mesh);

  atlasInterface::Field<double> in_v = atlas::array::make_view<double, 2>(in);
  atlasInterface::Field<double> out_v = atlas::array::make_view<double, 2>(out);

  for(int edge_idx = 0; edge_idx < mesh.edges().size(); ++edge_idx)
    in_v(edge_idx, 0) = 1.0;

  dawn_generated::cxxnaiveico::copyEdge<atlasInterface::atlasTag>(mesh, static_cast<int>(nb_levels),
                                                                  in_v, out_v)
      .run();

  for(int edge_idx = 0; edge_idx < mesh.edges().size(); ++edge_idx)
    ASSERT_EQ(out_v(edge_idx, 0), 1.0);
}
} // namespace

#include <generated_accumulateEdgeToCell.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, Accumulate) {
  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  size_t nb_levels = 1;

  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field in{fs_edges.createField<double>(atlas::option::name("in"))};

  atlas::functionspace::CellColumns fs(mesh, atlas::option::levels(nb_levels));
  atlas::Field out{fs.createField<double>(atlas::option::name("out"))};

  atlas::mesh::actions::build_edges(mesh);
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);

  atlasInterface::Field<double> in_v = atlas::array::make_view<double, 2>(in);
  atlasInterface::Field<double> out_v = atlas::array::make_view<double, 2>(out);

  for(int edge_idx = 0; edge_idx < mesh.edges().size(); ++edge_idx)
    in_v(edge_idx, 0) = 1.0;

  dawn_generated::cxxnaiveico::accumulateEdgeToCell<atlasInterface::atlasTag>(
      mesh, static_cast<int>(nb_levels), in_v, out_v)
      .run();

  for(int cell_idx = 0; cell_idx < mesh.cells().size(); ++cell_idx)
    ASSERT_EQ(out_v(cell_idx, 0), 4.0);
}
} // namespace

#include <generated_diffusion.hpp>
#include <reference_diffusion.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, Diffusion) {
  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  size_t nb_levels = 1;

  atlas::functionspace::CellColumns fs(mesh, atlas::option::levels(nb_levels));
  atlas::Field out_ref{fs.createField<double>(atlas::option::name("out"))};
  atlas::Field out_gen{fs.createField<double>(atlas::option::name("out"))};

  atlas::Field in_ref{fs.createField<double>(atlas::option::name("in"))};
  atlas::Field in_gen{fs.createField<double>(atlas::option::name("in"))};

  atlas::mesh::actions::build_edges(mesh);
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);

  {
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
      out_v_ref(jCell, 0) = 0.0;
      out_v_gen(jCell, 0) = 0.0;
    }
  }

  for(int i = 0; i < 500; ++i) {

    atlasInterface::Field<double> in_v_ref = atlas::array::make_view<double, 2>(in_ref);
    atlasInterface::Field<double> in_v_gen = atlas::array::make_view<double, 2>(in_gen);
    atlasInterface::Field<double> out_v_ref = atlas::array::make_view<double, 2>(out_ref);
    atlasInterface::Field<double> out_v_gen = atlas::array::make_view<double, 2>(out_gen);

    dawn_generated::cxxnaiveico::reference_diffusion<atlasInterface::atlasTag>(
        mesh, static_cast<int>(nb_levels), in_v_ref, out_v_ref)
        .run();
    dawn_generated::cxxnaiveico::diffusion<atlasInterface::atlasTag>(
        mesh, static_cast<int>(nb_levels), in_v_gen, out_v_gen)
        .run();

    using std::swap;
    swap(in_ref, out_ref);
    swap(in_gen, out_gen);
  }

  {
    auto out_v_ref = atlas::array::make_view<double, 2>(out_ref);
    auto out_v_gen = atlas::array::make_view<double, 2>(out_gen);
    AtlasVerifyer v;
    EXPECT_TRUE(v.compareArrayView(out_v_gen, out_v_ref)) << "while comparing output (on cells)";
  }
}
} // namespace
