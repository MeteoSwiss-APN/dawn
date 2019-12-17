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

#include "generated_quadGradient.hpp"

int main() {
  // kept low for now to get easy debug-able output
  const int numCell = 10;

  // apparently, one needs to be added to the second dimension in order to get a
  // square mesh, or we are mis-interpreting the output
  atlas::StructuredGrid structuredGrid =
      atlas::Grid("L" + std::to_string(numCell) + "x" + std::to_string(numCell + 1));
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  int nb_levels = 1;
  atlas::functionspace::CellColumns fs_cells(mesh, atlas::option::levels(nb_levels));
  atlas::Field f_cells{fs_cells.createField<double>(atlas::option::name("out"))};

  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field f_edges{fs_edges.createField<double>(atlas::option::name("in"))};

  atlas::mesh::actions::build_periodic_boundaries(mesh);
  atlas::mesh::actions::build_edges(mesh);
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
  atlas::mesh::actions::build_element_to_edge_connectivity(mesh);

  // does not change the neighborhood tables but probably manipulates something on the
  // halo / MPI level
  atlas::mesh::actions::build_periodic_boundaries(mesh);

  atlas::output::Gmsh gmesh("mymesh.msh");
  gmesh.write(mesh);

  atlasInterface::Field<double> v_cells = atlas::array::make_view<double, 2>(f_cells);
  atlasInterface::Field<double> v_edges = atlas::array::make_view<double, 2>(f_edges);

  {
    auto const& node_connectivity = mesh.cells().node_connectivity();
    auto lonlat = atlas::array::make_view<double, 2>(mesh.nodes().lonlat());

    // longitudes [0, 360], latitudes [-72 to 90 (?!)]
    auto lon0 = fmin(lonlat(node_connectivity(0, 0), 0),
                     lonlat(node_connectivity(mesh.cells().size() - 1, 0), 0));
    auto lat0 = fmin(lonlat(node_connectivity(0, 0), 1),
                     lonlat(node_connectivity(mesh.cells().size() - 1, 0), 1));
    auto lon1 = fmax(lonlat(node_connectivity(0, 0), 0),
                     lonlat(node_connectivity(mesh.cells().size() - 1, 0), 0));
    auto lat1 = fmax(lonlat(node_connectivity(0, 0), 1),
                     lonlat(node_connectivity(mesh.cells().size() - 1, 0), 1));
    for(int cell = 0, size = mesh.cells().size(); cell < size; ++cell) {
      double llx = lonlat(node_connectivity(cell, 0), 0);
      double lly = lonlat(node_connectivity(cell, 0), 1);
      double cartx = (llx - lon0) / (lon1 - lon0) * M_PI;
      double carty = (lly - lat0) / (lat1 - lat0) * M_PI;
      double val = sin(cartx) * sin(carty); // periodic signla fitting periodic boundaries
      v_cells(cell, 0) = val;
    }
  }

  gmesh.write(f_cells);

  // not running as expected yet. problem at boundaries (conceptual)
  dawn_generated::cxxnaiveico::gradient<atlasInterface::atlasTag>(mesh, nb_levels, v_cells, v_edges)
      .run();

  gmesh.write(f_cells);
}
