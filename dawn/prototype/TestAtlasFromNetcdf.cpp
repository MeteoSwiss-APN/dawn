//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include <assert.h>
#include <iostream>
#include <optional>
#include <string>

#include <atlas/library/Library.h>
#include <atlas/mesh/HybridElements.h>
#include <atlas/mesh/Mesh.h>
#include <atlas/mesh/Nodes.h>
#include <atlas/mesh/actions/BuildEdges.h>
#include <atlas/output/Gmsh.h>
#include <atlas/util/CoordinateEnums.h>

#include "AtlasFromNetcdf.h"

void debugDump(const atlas::Mesh& mesh) {
  auto lonlat = atlas::array::make_view<double, 2>(mesh.nodes().lonlat());
  const atlas::mesh::HybridElements::Connectivity& node_connectivity =
      mesh.cells().node_connectivity();

  {
    FILE* fp = fopen("netcdfMeshT.txt", "w+");
    for(int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {
      int nodeIdx0 = node_connectivity(cellIdx, 0) + 1;
      int nodeIdx1 = node_connectivity(cellIdx, 1) + 1;
      int nodeIdx2 = node_connectivity(cellIdx, 2) + 1;
      fprintf(fp, "%d %d %d\n", nodeIdx0, nodeIdx1, nodeIdx2);
    }
    fclose(fp);
  }

  {
    auto latToRad = [](double rad) { return rad / 90. * (0.5 * M_PI); };
    auto lonToRad = [](double rad) { return rad / 180. * M_PI; };
    FILE* fp = fopen("netcdfMeshP.txt", "w+");
    for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
      double latRad = latToRad(lonlat(nodeIdx, atlas::LAT));
      double lonRad = lonToRad(lonlat(nodeIdx, atlas::LON));

      const double R = 6371; // radius earth
      double x = R * cos(latRad) * cos(lonRad);
      double y = R * cos(latRad) * sin(lonRad);
      double z = R * sin(latRad);
      fprintf(fp, "%f %f %f\n", x, y, z);
    }
    fclose(fp);
  }

  // visualize with
  // P = load('netcdfMeshP.txt');
  // T = load('netcdfMeshT.txt');
  // trisurf(T(1:10,:),P(:,1),P(:,2),P(:,3))
}

int main(int argc, char const* argv[]) {
  if(argc != 2) {
    std::cout << "intended use is\n" << argv[0] << " input_file.nc" << std::endl;
    return -1;
  }
  std::string inFname(argv[1]);

  // version that uses atlas to compute additional neighbor lists
  {
    auto meshOpt = AtlasMeshFromNetCDFMinimal(inFname);
    assert(meshOpt.has_value());
    auto mesh = meshOpt.value();

    atlas::mesh::actions::build_edges(mesh, atlas::util::Config("pole_edges", false));
    atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
    atlas::mesh::actions::build_element_to_edge_connectivity(mesh);

    // dump mesh for Octave / MATLAB visualization (completely optional)
    debugDump(mesh);

    // dumo using the atlas gmsh writer
    atlas::output::Gmsh gmsh("earthV1.msh", atlas::util::Config("coordinates", "xyz"));
    gmsh.write(mesh);

    atlas::Library::instance().finalise();
    std::cout << "ran minimal version sucesfully!\n";
  }

  // version that reads additional neighbor lists from the netcdf file
  {
    auto meshOpt = AtlasMeshFromNetCDFComplete(inFname);
    assert(meshOpt.has_value());
    auto mesh = meshOpt.value();

    // dumo using the atlas gmsh writer
    atlas::output::Gmsh gmsh("earthV2.msh", atlas::util::Config("coordinates", "xyz"));
    gmsh.write(mesh);

    atlas::Library::instance().finalise();
    std::cout << "ran complete version sucesfully!\n";
  }
}
