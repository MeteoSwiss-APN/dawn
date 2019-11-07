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

#include "atlas/functionspace/CellColumns.h"
#include "atlas/functionspace/EdgeColumns.h"
#include "atlas/grid.h"
#include "atlas/mesh/actions/BuildCellCentres.h"
#include "atlas/mesh/actions/BuildEdges.h"
#include "atlas/meshgenerator.h"
#include "atlas/option/Options.h"
#include "atlas/output/Gmsh.h"
#include "dawn/CodeGen/Atlas/atlas_interface.hpp"

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
    in_v(cell_idx) = 1.0;

  dawn_generated::cxxnaiveico::copyCell<atlasInterface::atlasTag>(mesh, in_v, out_v).run();

  for(int cell_idx = 0; cell_idx < mesh.cells().size(); ++cell_idx)
    ASSERT_EQ(out_v(cell_idx), 1.0);
}
} // namespace

#include <generated_copyEdge.hpp>
namespace {
// TODO: this is currently broken, because cannot construct with IIRBuilder a stage with location
// type different from cells
TEST(AtlasIntegrationTestCompareOutput, DISABLED_CopyEdge) {
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
    in_v(edge_idx) = 1.0;

  dawn_generated::cxxnaiveico::copyEdge<atlasInterface::atlasTag>(mesh, in_v, out_v).run();

  for(int edge_idx = 0; edge_idx < mesh.edges().size(); ++edge_idx)
    ASSERT_EQ(out_v(edge_idx), 1.0);
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
    in_v(edge_idx) = 1.0;

  dawn_generated::cxxnaiveico::accumulateEdgeToCell<atlasInterface::atlasTag>(mesh, in_v, out_v)
      .run();

  for(int cell_idx = 0; cell_idx < mesh.cells().size(); ++cell_idx)
    ASSERT_EQ(out_v(cell_idx), 4.0);
}
} // namespace

#include <generated_diffusion.hpp>
#include <reference_diffusion.hpp>
namespace {
TEST(AtlasIntegrationTestCompareOutput, Diffusion) {}
} // namespace
