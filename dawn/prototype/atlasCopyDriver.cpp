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

#include "generated_copyEdgeToCell.hpp"

int main() {

  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);
  atlas::mesh::actions::build_edges(mesh);
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);

  int nb_levels = 1;
  atlas::functionspace::CellColumns fs(mesh, atlas::option::levels(nb_levels));
  atlas::Field out{fs.createField<double>(atlas::option::name("out"))};

  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field in{fs_edges.createField<double>(atlas::option::name("in"))};

  atlas::output::Gmsh gmesh("mymesh.msh");
  gmesh.write(mesh);

  for(int i = 0; i < 10; ++i) {
    in.metadata().set("step", i);
    gmesh.write(out);

    atlasInterface::Field<double> in_v = atlas::array::make_view<double, 2>(in);
    atlasInterface::Field<double> out_v = atlas::array::make_view<double, 2>(out);

    dawn_generated::cxxnaiveico::generated<atlasInterface::atlasTag>(mesh, nb_levels, in_v, out_v)
        .run();
  }
}
