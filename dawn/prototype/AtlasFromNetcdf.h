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

#include <optional>
#include <string>

#include <atlas/mesh/Mesh.h>

// This module offers facilities to load a dwd netcdf file (*.nc) into an Atlas mesh
//
// - This is not intended for general netcdf files, but only meshes as obtained from the dwd web
//   service: https://oflxd21.dwd.de/cgi-bin/spp1167/webservice.cgi. Consequently, if the file does
//   not conform to the dwd naming conventions (e.g. the neighbor tables are not named with the same
//   scheme), an error message is printed and std::nullopt is returned
//
// - Two different readers are offered: the minimal version only reads the node locations and cell
//   to node neighbor table ("vertex_of_cell"). If required, additional neighbor tables can then be
//   computed using the atlas actions if required:
//          atlas::mesh::actions::build_edges(mesh, atlas::util::Config("pole_edges", false));
//          atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
//          atlas::mesh::actions::build_element_to_edge_connectivity(mesh);
//   The complete version deserializes all neighborhood tables that are both supported by atlas and
//   present in netcdf. These are all possible tables ecept node to node (not present in Atlas) and
//   edge to edge (not present in the DWD netcdf files). Using this reader ensures that all
//   neighbors are encountered in the same sequence as in the ICON fortran code
//
// - Both readers only generate one Atlas partition. This also means that NO halos are generated and
//   MPI will NOT work!
//
// - The reader also does not assign xy values since such a mapping would require a map projection,
//   where no universally accepted default exists (c.f.
//   https://en.wikipedia.org/wiki/Map_projection)
//

std::optional<atlas::Mesh> AtlasMeshFromNetCDFMinimal(const std::string& filename);
std::optional<atlas::Mesh> AtlasMeshFromNetCDFComplete(const std::string& filename);