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
#include "generated_quadGradient.hpp"

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

  AtlasToCartesian atlasToCartesianMapper(mesh);
  build_periodic_edges(mesh, numCell, numCell, atlasToCartesianMapper);

  int nb_levels = 1;
  atlas::functionspace::CellColumns fs_cells(mesh, atlas::option::levels(nb_levels));
  atlas::Field f_cells{fs_cells.createField<double>(atlas::option::name("out"))};

  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field f_edges{fs_edges.createField<double>(atlas::option::name("in"))};

  atlasInterface::Field<double> v_cells = atlas::array::make_view<double, 2>(f_cells);
  atlasInterface::Field<double> v_edges = atlas::array::make_view<double, 2>(f_edges);

  for(int cellIdx = 0, size = mesh.cells().size(); cellIdx < size; ++cellIdx) {
    auto [cartX, cartY] = atlasToCartesianMapper.cellMidpoint(mesh, cellIdx);
    double val =
        sin(cartX * M_PI) * sin(cartY * M_PI); // periodic signal fitting periodic boundaries
    v_cells(cellIdx, 0) = val;
  }

  atlas::output::Gmsh gmesh("mymesh.msh");
  gmesh.write(mesh);
  gmesh.write(f_cells);

  // not running as expected yet. problem at boundaries (conceptual)
  dawn_generated::cxxnaiveico::gradient<atlasInterface::atlasTag>(mesh, nb_levels, v_cells, v_edges)
      .run();

  gmesh.write(f_cells);
}
