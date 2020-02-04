#include <atlas/array/ArrayShape.h>
#include <fstream>

#include "atlas/functionspace/CellColumns.h"
#include "atlas/functionspace/EdgeColumns.h"
#include "atlas/grid.h"
#include "atlas/mesh/actions/BuildCellCentres.h"
#include "atlas/mesh/actions/BuildEdges.h"
#include "atlas/meshgenerator.h"
#include "atlas/option/Options.h"
#include "atlas/output/Gmsh.h"
#include "atlas_interface.hpp"

#include "AtlasCartesianWrapper.h"
#include "generated_sparseDimension.hpp"

int main() {

  atlas::StructuredGrid structuredGrid = atlas::Grid("L10x11");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);
  atlas::mesh::actions::build_edges(mesh, atlas::util::Config("pole_edges", false));
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
  atlas::mesh::actions::build_element_to_edge_connectivity(mesh);

  AtlasToCartesian atlasToCartesianMapper(mesh);

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
      int edgeIdx = mesh.cells().edge_connectivity()(iCell, jNbh);
      auto [x, y] = atlasToCartesianMapper.edgeMidpoint(mesh, edgeIdx);
      sparseDim_v(iCell, jNbh, level) = x * x + y * y;
    }
  }

  for(int iEdge = 0; iEdge < mesh.edges().size(); iEdge++) {
    edges_v(iEdge, level) = 1;
  }

  dawn_generated::cxxnaiveico::sparseDimension<atlasInterface::atlasTag>(mesh, nb_levels, cells_v,
                                                                         edges_v, sparseDim_v)
      .run();

  FILE* fp = fopen("sparseDimAtlas.txt", "w+");
  for(int iCell = 0; iCell < mesh.cells().size(); iCell++) {
    auto [x, y] = atlasToCartesianMapper.cellMidpoint(mesh, iCell);
    fprintf(fp, "%f %f %f\n", x, y, cells_v(level, iCell));
  }
  fclose(fp);

  // visualize in octave for a nice color gradient:
  // p = load('sparseDimAtlas.txt')
  // scatter(p(:,1),p(:,2),50,p(:,3),'filled');
}
