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
#include "atlas_interface.hpp"

#include "generated_verticalSum.hpp"

int main() {

  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  int nb_levels = 5;
  atlas::functionspace::CellColumns fs(mesh, atlas::option::levels(nb_levels));
  atlas::Field out{fs.createField<double>(atlas::option::name("out"))};

  atlas::functionspace::CellColumns fs_edges(mesh, atlas::option::levels(nb_levels));
  atlas::Field in{fs_edges.createField<double>(atlas::option::name("in"))};

  atlas::output::Gmsh gmesh("mymesh.msh");

  atlasInterface::Field<double> in_v = atlas::array::make_view<double, 2>(in);
  atlasInterface::Field<double> out_v = atlas::array::make_view<double, 2>(out);

  for(int level = 0; level < nb_levels; ++level) {
    for(int cell = 0; cell < mesh.cells().size(); ++cell) {
      in_v(cell, level) = 10;
    }
  }

  dawn_generated::cxxnaiveico::generated<atlasInterface::atlasTag>(mesh, nb_levels, in_v, out_v)
      .run();

  gmesh.write(out);
}
