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

#include "generated/generated_Diffusion.hpp"
#include "reference/reference_Diffusion.hpp"

#include "atlasVerifyer.h"

int main() {

  atlas::StructuredGrid structuredGrid = atlas::Grid("L32x32");
  atlas::StructuredMeshGenerator generator;
  auto mesh = generator.generate(structuredGrid);

  int nb_levels = 1;
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

  atlas::output::Gmsh gmesh("mymesh.msh");
  gmesh.write(mesh);

  for(int i = 0; i < 500; ++i) {
    out_gen.metadata().set("step", i);
    gmesh.write(out_gen);

    atlasInterface::Field<double> in_v_ref = atlas::array::make_view<double, 2>(in_ref);
    atlasInterface::Field<double> in_v_gen = atlas::array::make_view<double, 2>(in_gen);
    atlasInterface::Field<double> out_v_ref = atlas::array::make_view<double, 2>(out_ref);
    atlasInterface::Field<double> out_v_gen = atlas::array::make_view<double, 2>(out_gen);

    dawn_generated::cxxnaiveico::generated<atlasInterface::atlasTag>(mesh, in_v_ref, out_v_ref)
        .run();
    dawn_generated::cxxnaiveico::generated<atlasInterface::atlasTag>(mesh, in_v_gen, out_v_gen)
        .run();

    using std::swap;
    swap(in_ref, out_ref);
    swap(in_gen, out_gen);
  }

  {
    auto out_v_ref = atlas::array::make_view<double, 2>(out_ref);
    auto out_v_gen = atlas::array::make_view<double, 2>(out_gen);
    atlasVerifyer v;
    assert(v.compareArrayView(out_v_gen, out_v_ref));
  }
}
