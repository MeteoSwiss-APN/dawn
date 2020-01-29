#include <fstream>

#include "atlas/functionspace/CellColumns.h"
#include "atlas/functionspace/EdgeColumns.h"
#include "atlas/grid.h"
#include "atlas/mesh/actions/BuildCellCentres.h"
#include "atlas/mesh/actions/BuildEdges.h"
#include "atlas/mesh/actions/BuildPeriodicBoundaries.h"
#include "atlas/meshgenerator.h"
#include "atlas/option/Options.h"
#include "atlas/output/Gmsh.h"
#include "atlas_interface.hpp"

#include "AtlasCartesianWrapper.h"
#include "generated_NestedWithField.hpp"

int main() {
  // kept low for now to get easy debug-able output
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

  dawn_generated::cxxnaiveico::nested<atlasInterface::atlasTag>(mesh, nb_levels, v_cells, v_edges,
                                                                v_vertices)
      .run();

  // each vertex stores value 1                 1
  // vertices are reduced onto edges            2
  // each edge stores 200                     202
  // each face reduces its edges (4 per face) 808
  for(int i = 0; i < mesh.cells().size(); i++) {
    assert(fabs(v_cells(i, 0) - 808) < 1e-12);
  }
}
