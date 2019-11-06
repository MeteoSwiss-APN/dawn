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

#include <generated_copyEdgeToCell.hpp>
#include <reference_copyEdgeToCell.hpp>

namespace {
TEST(AtlasIntegrationTestGen, CopyGen) {
  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  int nb_levels = 1;
  atlas::functionspace::CellColumns fs(mesh, atlas::option::levels(nb_levels));
  atlas::Field out{fs.createField<double>(atlas::option::name("out"))};

  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field in{fs_edges.createField<double>(atlas::option::name("in"))};

  atlas::mesh::actions::build_edges(mesh);
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);

  for(int i = 0; i < 10; ++i) {
    atlasInterface::Field<double> in_v = atlas::array::make_view<double, 2>(in);
    atlasInterface::Field<double> out_v = atlas::array::make_view<double, 2>(out);

    dawn_generated::cxxnaiveico::generated<atlasInterface::atlasTag>(mesh, in_v, out_v).run();
    dawn_generated::cxxnaiveico::reference<atlasInterface::atlasTag>(mesh, in_v, out_v).run();
  }

  // TODO generate ref
  // TODO compare with ref
}
} // namespace
