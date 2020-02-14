/* This is part of the netCDF package.
   Copyright 2006 University Corporation for Atmospheric Research/Unidata.
   See COPYRIGHT file for conditions of use.

   This is a very simple example which reads a 2D array of
   sample data produced by simple_xy_wr.cpp.

   This example is part of the netCDF tutorial:
   http://www.unidata.ucar.edu/software/netcdf/docs/netcdf-tutorial

   Full documentation of the netCDF C++ API can be found at:
   http://www.unidata.ucar.edu/software/netcdf/docs/netcdf-cxx

   $Id: simple_xy_rd.cpp,v 1.5 2010/02/11 22:36:43 russ Exp $
*/

#include <assert.h>
#include <iostream>
#include <math.h>
#include <optional>
#include <string>

#include <netcdf>

#include <atlas/array/Array.h>
#include <atlas/array/ArrayView.h>
#include <atlas/array/IndexView.h>
#include <atlas/library/Library.h>
#include <atlas/library/config.h>
#include <atlas/mesh/ElementType.h>
#include <atlas/mesh/Elements.h>
#include <atlas/mesh/HybridElements.h>
#include <atlas/mesh/Mesh.h>
#include <atlas/mesh/Nodes.h>
#include <atlas/mesh/actions/BuildCellCentres.h>
#include <atlas/mesh/actions/BuildEdges.h>
#include <atlas/mesh/actions/BuildPeriodicBoundaries.h>
#include <atlas/meshgenerator/detail/MeshGeneratorImpl.h>
#include <atlas/output/Gmsh.h>
#include <atlas/util/CoordinateEnums.h>

static const int myPart = 0;

template <typename loadT>
std::vector<loadT> LoadField(const netCDF::NcFile& dataFile, const std::string& name) {
  netCDF::NcVar data = dataFile.getVar(name.c_str());
  if(data.isNull()) {
    return {};
  }
  assert(data.getDimCount() == 1);
  int size = data.getDim(0).getSize();
  std::vector<loadT> ret(size);
  data.getVar(ret.data());
  return ret;
}

template <typename loadT>
std::tuple<std::vector<loadT>, int, int> Load2DField(const netCDF::NcFile& dataFile,
                                                     const std::string& name) {
  netCDF::NcVar data = dataFile.getVar(name.c_str());
  if(data.isNull()) {
    return {};
  }
  assert(data.getDimCount() == 2);
  int stride = data.getDim(0).getSize();
  int elPerStride = data.getDim(1).getSize();
  int size = stride * elPerStride;
  std::vector<loadT> ret(size);
  data.getVar(ret.data());
  return {ret, stride, elPerStride};
}

bool NodesFromNetCDF(const netCDF::NcFile& dataFile, atlas::Mesh& mesh) {
  auto lon = LoadField<double>(dataFile, "vlon");
  auto lat = LoadField<double>(dataFile, "vlat");
  if(lon.size() == 0 || lat.size() == 0) {
    std::cout << "lat / long variable not found\n";
    return false;
  }
  if(lon.size() != lat.size()) {
    std::cout << "lat / long not of consistent sizes!\n";
    return false;
  }

  int numNodes = lat.size();

  // define nodes and associated properties for Atlas meshs
  mesh.nodes().resize(numNodes);
  atlas::mesh::Nodes& nodes = mesh.nodes();
  auto xy = atlas::array::make_view<double, 2>(nodes.xy());
  auto lonlat = atlas::array::make_view<double, 2>(nodes.lonlat());

  // we currently don't care about parts, so myPart is always 0 and remotde_idx == glb_idx
  auto glb_idx_node = atlas::array::make_view<atlas::gidx_t, 1>(nodes.global_index());
  auto remote_idx = atlas::array::make_indexview<atlas::idx_t, 1>(nodes.remote_index());
  auto part = atlas::array::make_view<int, 1>(nodes.partition());

  // no ghosts currently (ghost = false always) and no flags are set
  auto ghost = atlas::array::make_view<int, 1>(nodes.ghost());
  auto flags = atlas::array::make_view<int, 1>(nodes.flags());

  auto radToLat = [](double rad) { return rad / (0.5 * M_PI) * 90; };
  auto radToLon = [](double rad) { return rad / (M_PI)*180; };

  for(int nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
    xy(nodeIdx, atlas::LON) = radToLon(lon[nodeIdx]);
    xy(nodeIdx, atlas::LAT) = radToLat(lat[nodeIdx]);

    lonlat(nodeIdx, atlas::LON) = xy(nodeIdx, atlas::LON);
    lonlat(nodeIdx, atlas::LAT) = xy(nodeIdx, atlas::LAT);

    glb_idx_node(nodeIdx) = nodeIdx;
    remote_idx(nodeIdx) = nodeIdx;

    part(nodeIdx) = myPart;
    ghost(nodeIdx) = false;
    atlas::mesh::Nodes::Topology::reset(flags(nodeIdx));
  }

  return true;
}

bool CellsFromNetCDF(const netCDF::NcFile& dataFile, atlas::Mesh& mesh) {
  auto [cellToVertex, vertexPerCell, ncells] = Load2DField<int>(dataFile, "vertex_of_cell");
  if(vertexPerCell != 3) {
    std::cout << "not a triangle mesh\n";
    return false;
  }

  // define cells and associated properties
  mesh.cells().add(new atlas::mesh::temporary::Triangle(), ncells);
  auto cells_part = atlas::array::make_view<int, 1>(mesh.cells().partition());
  atlas::mesh::HybridElements::Connectivity& node_connectivity = mesh.cells().node_connectivity();
  atlas::array::ArrayView<atlas::gidx_t, 1> glb_idx_cell =
      atlas::array::make_view<atlas::gidx_t, 1>(mesh.cells().global_index());

  for(int cellIdx = 0; cellIdx < ncells; cellIdx++) { // indices in netcdf are 1 based
    atlas::idx_t tri_nodes[3] = {tri_nodes[0] = cellToVertex[0 * ncells + cellIdx] - 1,
                                 tri_nodes[1] = cellToVertex[1 * ncells + cellIdx] - 1,
                                 tri_nodes[2] = cellToVertex[2 * ncells + cellIdx] - 1};
    node_connectivity.set(cellIdx, tri_nodes);
    glb_idx_cell[cellIdx] = cellIdx;
    cells_part(cellIdx) = myPart;
  }

  return true;
}

std::optional<atlas::Mesh> AtlasMeshFromNetcdf(const std::string& filename) {
  try {
    atlas::Mesh mesh;
    netCDF::NcFile dataFile(filename.c_str(), netCDF::NcFile::read);

    if(!NodesFromNetCDF(dataFile, mesh)) {
      return {};
    }

    if(!CellsFromNetCDF(dataFile, mesh)) {
      return {};
    }

    return mesh;
  } catch(netCDF::exceptions::NcException& e) {
    std::cout << e.what() << "\n";
    return std::nullopt;
  }
}

void debugDump(const atlas::Mesh& mesh) {
  auto lonlat = atlas::array::make_view<double, 2>(mesh.nodes().lonlat());
  const atlas::mesh::HybridElements::Connectivity& node_connectivity =
      mesh.cells().node_connectivity();

  {
    FILE* fp = fopen("netcdfMeshT.txt", "w+");
    for(int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {
      int nodeIdx0 = node_connectivity(cellIdx, 0);
      int nodeIdx1 = node_connectivity(cellIdx, 1);
      int nodeIdx2 = node_connectivity(cellIdx, 2);
      fprintf(fp, "%d %d %d\n", nodeIdx0 + 1, nodeIdx1 + 1, nodeIdx2 + 1);
    }
    fclose(fp);
  }

  {
    FILE* fp = fopen("netcdfMeshP.txt", "w+");
    for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
      double lat = lonlat(nodeIdx, atlas::LAT);
      double lon = lonlat(nodeIdx, atlas::LON);

      const double R = 6371; // radius earth
      double x = R * cos(lat) * cos(lon);
      double y = R * cos(lat) * sin(lon);
      double z = R * sin(lat);
      fprintf(fp, "%f %f %f\n", x, y, z);
    }
    fclose(fp);
  }
}

int main() {
  // Open the file for read access
  auto meshOpt = AtlasMeshFromNetcdf("icon_grid_0010_R02B04_G.nc");
  assert(meshOpt.has_value());
  auto mesh = meshOpt.value();

  atlas::mesh::actions::build_edges(mesh, atlas::util::Config("pole_edges", false));
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
  atlas::mesh::actions::build_element_to_edge_connectivity(mesh);

  debugDump(mesh);

  atlas::output::Gmsh gmsh("earth.msh", atlas::util::Config("coordinates", "xyz"));
  gmsh.write(mesh);

  atlas::Library::instance().finalise();

  printf("done!\n");
  return 0;
}
